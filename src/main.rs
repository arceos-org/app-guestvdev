//! ArceOS Guest Virtual Device (Hypervisor)
//!
//! Derived from the h_3_0 tutorial crate in the ArceOS ecosystem.
//! Runs a guest OS with virtual device support (timer virtualization,
//! console I/O, nested page fault passthrough) on RISC-V H-extension,
//! ARM AArch64, and AMD SVM.
//!
//! The h_3_0 control flow:
//!   1. Create guest address space with pre-allocated RAM
//!   2. Load guest binary from filesystem
//!   3. Setup vCPU context
//!   4. Run guest in loop, handling VM exits:
//!      - SBI/Hypercall forwarding (PutChar, SetTimer, Shutdown, etc.)
//!      - Nested page fault → passthrough mapping
//!      - Timer interrupt → inject to guest (for preemptive scheduling)

#![cfg_attr(feature = "axstd", no_std)]
#![cfg_attr(feature = "axstd", no_main)]
#![cfg_attr(all(feature = "axstd", target_arch = "riscv64"), feature(riscv_ext_intrinsics))]

#[cfg(feature = "axstd")]
extern crate axstd as std;

#[cfg(feature = "axstd")]
extern crate alloc;

#[cfg(feature = "axstd")]
#[macro_use]
extern crate axlog;

#[cfg(feature = "axstd")]
extern crate axfs;
#[cfg(feature = "axstd")]
extern crate axio;

// ────────────────── RISC-V 64 specific modules ──────────────────
#[cfg(all(feature = "axstd", target_arch = "riscv64"))]
mod vcpu;
#[cfg(all(feature = "axstd", target_arch = "riscv64"))]
mod regs;
#[cfg(all(feature = "axstd", target_arch = "riscv64"))]
mod csrs;
#[cfg(all(feature = "axstd", target_arch = "riscv64"))]
mod sbi;

// ────────────────── AArch64 specific modules ──────────────────
// NOTE: The current AArch64 approach uses a bootloader-style handoff
// (hypervisor loads and jumps to guest at EL1), so the legacy EL1→EL0
// guest/host context-switching modules are not needed.
// The source files in src/aarch64/ are preserved for reference.
// #[cfg(all(feature = "axstd", target_arch = "aarch64"))]
// #[path = "aarch64/mod.rs"]
// mod aarch64;

// ────────────────── x86_64 (AMD SVM) specific modules ──────────────────
#[cfg(all(feature = "axstd", target_arch = "x86_64"))]
#[path = "x86_64/mod.rs"]
mod x86_64_svm;

// ────────────────── Common modules ──────────────────
#[cfg(feature = "axstd")]
mod loader;

// VM entry point (guest physical / intermediate-physical address)
#[cfg(all(feature = "axstd", target_arch = "riscv64"))]
const VM_ENTRY: usize = 0x8020_0000;

#[cfg(all(feature = "axstd", target_arch = "aarch64"))]
const VM_ENTRY: usize = 0x4420_0000;

#[cfg(all(feature = "axstd", target_arch = "x86_64"))]
const VM_ENTRY: usize = 0x10000;

#[cfg(all(
    feature = "axstd",
    not(any(target_arch = "riscv64", target_arch = "aarch64", target_arch = "x86_64"))
))]
const VM_ENTRY: usize = 0x8020_0000;

// ════════════════════════════════════════════════════════════════
//  Entry point
// ════════════════════════════════════════════════════════════════

#[cfg_attr(feature = "axstd", unsafe(no_mangle))]
fn main() {
    #[cfg(all(feature = "axstd", target_arch = "riscv64"))]
    riscv64_main();

    #[cfg(all(feature = "axstd", target_arch = "aarch64"))]
    aarch64_main();

    #[cfg(all(feature = "axstd", target_arch = "x86_64"))]
    x86_64_main();

    #[cfg(not(feature = "axstd"))]
    {
        println!("This application requires the 'axstd' feature for running the Hypervisor.");
        println!("Run with: cargo xtask run [--arch riscv64|aarch64|x86_64]");
    }
}

// ════════════════════════════════════════════════════════════════
//  RISC-V 64  (H-extension hypervisor — h_3_0 style)
//
//  Full OS guest support with virtual device handling:
//  - Timer virtualization (SetTimer + hvip injection) for preemptive
//    scheduling in the guest
//  - Console I/O via SBI PutChar/GetChar forwarding
//  - NPF passthrough for MMIO devices
//
//  The guest runs u_6_0-style multi-tasking demo with CFS scheduler.
// ════════════════════════════════════════════════════════════════

#[cfg(all(feature = "axstd", target_arch = "riscv64"))]
fn riscv64_main() {
    use alloc::sync::Arc;
    use vcpu::VmCpuRegisters;
    use riscv::register::scause;
    use csrs::defs::hstatus;
    use csrs::traps;
    use tock_registers::LocalRegisterCopy;
    use csrs::{RiscvCsrTrait, CSR};
    use vcpu::_run_guest;
    use axhal::mem::PhysAddr;
    use axhal::paging::{MappingFlags, PageSize};
    use axmm::backend::{Backend, SharedPages};
    use memory_addr::{va, PAGE_SIZE_4K};

    ax_println!("Starting virtualization...");

    // ════════════════════════════════════════════════════
    //  Step 0: Setup H-extension CSRs  (matches h_3_0 riscv_vcpu::setup_csrs)
    // ════════════════════════════════════════════════════
    unsafe {
        // Delegate VS-mode synchronous exceptions to the guest so it can
        // handle its own page faults, illegal instructions, breakpoints, etc.
        CSR.hedeleg.write_value(
            traps::exception::INST_ADDR_MISALIGN
                | traps::exception::BREAKPOINT
                | traps::exception::ENV_CALL_FROM_U_OR_VU
                | traps::exception::INST_PAGE_FAULT
                | traps::exception::LOAD_PAGE_FAULT
                | traps::exception::STORE_PAGE_FAULT
                | traps::exception::ILLEGAL_INST,
        );

        // Delegate VS-mode interrupts to the guest.
        CSR.hideleg.write_value(
            traps::interrupt::VIRTUAL_SUPERVISOR_TIMER
                | traps::interrupt::VIRTUAL_SUPERVISOR_EXTERNAL
                | traps::interrupt::VIRTUAL_SUPERVISOR_SOFT,
        );

        // Clear all pending virtual interrupts.
        CSR.hvip.read_and_clear_bits(
            traps::interrupt::VIRTUAL_SUPERVISOR_TIMER
                | traps::interrupt::VIRTUAL_SUPERVISOR_EXTERNAL
                | traps::interrupt::VIRTUAL_SUPERVISOR_SOFT,
        );

        // Allow the guest to read all counters (cycle, time, instret, HPMs).
        CSR.hcounteren.write_value(0xffff_ffff);

        // Clear SIE timer bit — we will enable it when the guest calls SetTimer.
        CSR.sie
            .read_and_clear_bits(traps::interrupt::SUPERVISOR_TIMER);
    }

    // ════════════════════════════════════════════════════
    //  Step 1: Create guest address space (h_3_0: AddrSpace::new_empty)
    // ════════════════════════════════════════════════════
    let mut uspace = axmm::AddrSpace::new_empty(va!(0x0), 0x7fff_ffff_f000).unwrap();

    let flags = MappingFlags::READ | MappingFlags::WRITE
        | MappingFlags::EXECUTE | MappingFlags::USER;

    // ════════════════════════════════════════════════════
    //  Step 2: Pre-allocate guest physical RAM
    //
    //  h_3_0: map_alloc(0x8000_0000, 0x100_0000, flags, true)
    //  Pre-allocate 16MB at 0x8000_0000 to avoid NPF during guest boot.
    // ════════════════════════════════════════════════════
    const PHY_MEM_START: usize = 0x8000_0000;
    const PHY_MEM_SIZE: usize = 0x100_0000; // 16 MB

    ax_println!("Pre-allocating {} MB guest RAM at {:#x}...", PHY_MEM_SIZE / (1024 * 1024), PHY_MEM_START);
    let pages = Arc::new(
        SharedPages::new(PHY_MEM_SIZE, PageSize::Size4K)
            .expect("alloc guest RAM pages"),
    );
    uspace
        .map(
            PHY_MEM_START.into(),
            PHY_MEM_SIZE,
            flags,
            true,
            Backend::new_shared(PHY_MEM_START.into(), pages),
        )
        .expect("map guest RAM");

    // ════════════════════════════════════════════════════
    //  Step 3: Load guest binary into pre-allocated RAM
    //
    //  h_3_0 loads from /sbin/u_6_0_riscv64-qemu-virt.bin.
    //  We use the unified guest payload /sbin/gkernel.
    // ════════════════════════════════════════════════════
    {
        let fname = "/sbin/gkernel";
        ax_println!("VM created success, loading images...");
        ax_println!("app: {}", fname);
        let ctx = axfs::ROOT_FS_CONTEXT.get().expect("Root FS not initialized");
        let file = axfs::File::open(ctx, fname).expect("Cannot open guest image");
        let mut offset = 0usize;
        let mut total_bytes = 0usize;
        loop {
            let mut buf = [0u8; 4096];
            let n = axio::Read::read(&mut &file, &mut buf).expect("read");
            if n == 0 {
                break;
            }
            total_bytes += n;
            uspace
                .write((VM_ENTRY + offset).into(), &buf[..n])
                .expect("write guest image");
            offset += n;
            if n < 4096 {
                break;
            }
        }
        ax_println!("Loaded {} bytes from {}", total_bytes, fname);
    }

    // ════════════════════════════════════════════════════
    //  Step 4: Prepare guest context & G-stage page table
    //  (h_3_0: arch_vcpu.set_entry / arch_vcpu.set_ept_root)
    // ════════════════════════════════════════════════════
    let mut ctx = VmCpuRegisters::default();
    prepare_guest_context(&mut ctx);

    let ept_root = uspace.page_table_root();
    ax_println!("bsp_entry: {:#x}; ept: {:#x}", VM_ENTRY, ept_root);
    prepare_vm_pgtable(ept_root);

    // ════════════════════════════════════════════════════
    //  Step 5: Run guest in loop  (h_3_0 style)
    //
    //  Handle:
    //    - VirtualSupervisorEnvCall (scause 10): SBI calls
    //      (PutChar, SetTimer, Shutdown, etc.)
    //    - Guest page faults (scause 20/21/23): MMIO passthrough
    //    - Supervisor timer interrupt: inject to guest via hvip
    //      (required for guest preemptive multitasking)
    // ════════════════════════════════════════════════════
    ax_println!("Entering VM run loop...");

    loop {
        // Disable host interrupts while guest is running (like h_3_0 vcpu_run)
        let saved_sstatus: usize;
        unsafe {
            core::arch::asm!("csrrci {}, sstatus, 0x2", out(reg) saved_sstatus);
            _run_guest(&mut ctx);
            core::arch::asm!("csrs sstatus, {}", in(reg) saved_sstatus & 0x2);
        }

        let scause = scause::read();

        // ── Interrupts ──
        if scause.is_interrupt() {
            match scause.code() {
                5 => {
                    // SupervisorTimer: inject virtual timer interrupt to guest
                    // (required for guest preemptive multitasking — CFS scheduler)
                    CSR.hvip
                        .read_and_set_bits(traps::interrupt::VIRTUAL_SUPERVISOR_TIMER);
                    // Disable host timer until guest re-arms it via SetTimer
                    CSR.sie
                        .read_and_clear_bits(traps::interrupt::SUPERVISOR_TIMER);
                }
                _ => {}
            }
            continue;
        }

        // ── Exceptions ──
        match scause.code() {
            10 => {
                // VirtualSupervisorEnvCall — SBI call from guest
                let a7 = ctx.guest_regs.gprs.a_regs()[7]; // extension ID
                let a6 = ctx.guest_regs.gprs.a_regs()[6]; // function ID

                // ── Shutdown ──
                if a7 == 8 {
                    ax_println!("Guest: SBI legacy shutdown");
                    break;
                }
                if a7 == 0x53525354 {
                    ax_println!("Guest: SBI SRST shutdown");
                    break;
                }

                // ── Legacy SBI PutChar (fast path: write directly to UART) ──
                if a7 == 1 {
                    let ch = ctx.guest_regs.gprs.a_regs()[0] as u8;
                    let uart_va = axhal::mem::phys_to_virt(
                        PhysAddr::from(0x1000_0000usize),
                    ).as_usize();
                    unsafe {
                        core::ptr::write_volatile(uart_va as *mut u8, ch);
                    }
                    ctx.guest_regs.sepc += 4;
                    continue;
                }

                // ── SBI SetTimer (proper timer virtualization for preemptive scheduling) ──
                if a7 == 0x54494D45 || (a7 == 0 && a6 == 0) {
                    // TIME extension (EID 0x54494D45, FID 0) or legacy SetTimer (EID 0)
                    let timer_val = ctx.guest_regs.gprs.a_regs()[0];
                    sbi_rt::set_timer(timer_val as u64);
                    // Clear guest timer pending
                    CSR.hvip
                        .read_and_clear_bits(traps::interrupt::VIRTUAL_SUPERVISOR_TIMER);
                    // Enable host timer interrupt so we catch it
                    CSR.sie
                        .read_and_set_bits(traps::interrupt::SUPERVISOR_TIMER);
                    ctx.guest_regs.gprs.set_reg(regs::GprIndex::A0, 0);
                    ctx.guest_regs.sepc += 4;
                    continue;
                }

                // ── Legacy SBI GetChar ──
                if a7 == 2 {
                    #[allow(deprecated)]
                    let c = sbi_rt::legacy::console_getchar();
                    ctx.guest_regs.gprs.set_reg(regs::GprIndex::A0, c);
                    ctx.guest_regs.sepc += 4;
                    continue;
                }

                // ── Forward all other SBI calls to the real SBI (OpenSBI) ──
                let a0 = ctx.guest_regs.gprs.a_regs()[0];
                let a1 = ctx.guest_regs.gprs.a_regs()[1];
                let a2 = ctx.guest_regs.gprs.a_regs()[2];
                let a3 = ctx.guest_regs.gprs.a_regs()[3];
                let a4 = ctx.guest_regs.gprs.a_regs()[4];
                let a5 = ctx.guest_regs.gprs.a_regs()[5];

                let ret_error: usize;
                let ret_value: usize;
                unsafe {
                    core::arch::asm!(
                        "ecall",
                        inout("a0") a0 => ret_error,
                        inout("a1") a1 => ret_value,
                        in("a2") a2,
                        in("a3") a3,
                        in("a4") a4,
                        in("a5") a5,
                        in("a6") a6,
                        in("a7") a7,
                    );
                }
                ctx.guest_regs.gprs.set_reg(regs::GprIndex::A0, ret_error);
                ctx.guest_regs.gprs.set_reg(regs::GprIndex::A1, ret_value);
                ctx.guest_regs.sepc += 4;
            }

            20 | 21 | 23 => {
                // Guest page fault (G-stage) — MMIO passthrough
                // h_3_0 handles pflash at 0x2200_0000 with passthrough mode.
                // We use generic passthrough for any MMIO address.
                let htval: usize;
                let stval_val: usize;
                unsafe {
                    core::arch::asm!("csrr {}, htval", out(reg) htval);
                    core::arch::asm!("csrr {}, stval", out(reg) stval_val);
                }
                let fault_addr = (htval << 2) | (stval_val & 0x3);
                let page_addr = fault_addr & !0xFFF;

                // Passthrough-map for MMIO devices (pflash, etc.)
                // h_3_0: aspace.map_linear(addr, addr.as_usize().into(), 4096, mapping_flags)
                let _ = uspace.map_linear(
                    page_addr.into(),
                    PhysAddr::from(page_addr),
                    PAGE_SIZE_4K,
                    flags,
                );

                unsafe {
                    core::arch::riscv64::hfence_gvma_all();
                }
            }

            _ => {
                let stval_val: usize;
                let htval_val: usize;
                unsafe {
                    core::arch::asm!("csrr {}, stval", out(reg) stval_val);
                    core::arch::asm!("csrr {}, htval", out(reg) htval_val);
                }
                ax_println!(
                    "Unhandled trap: code={}, sepc={:#x}, stval={:#x}, htval={:#x}",
                    scause.code(),
                    ctx.guest_regs.sepc,
                    stval_val,
                    htval_val
                );
                break;
            }
        }
    }

    ax_println!("Shutdown vm normally!");
    panic!("Hypervisor ok!");

    fn prepare_vm_pgtable(ept_root: PhysAddr) {
        let hgatp = 8usize << 60 | usize::from(ept_root) >> 12;
        unsafe {
            core::arch::asm!(
                "csrw hgatp, {hgatp}",
                hgatp = in(reg) hgatp,
            );
            core::arch::riscv64::hfence_gvma_all();
        }
    }

    fn prepare_guest_context(ctx: &mut VmCpuRegisters) {
        use csrs::{RiscvCsrTrait, CSR};
        let hstatus_val: usize;
        unsafe {
            core::arch::asm!("csrr {}, hstatus", out(reg) hstatus_val);
        }
        let mut hstatus_reg = LocalRegisterCopy::<usize, hstatus::Register>::new(hstatus_val);
        hstatus_reg.modify(hstatus::spv::Guest);
        hstatus_reg.modify(hstatus::spvp::Supervisor);
        CSR.hstatus.write_value(hstatus_reg.get());
        ctx.guest_regs.hstatus = hstatus_reg.get();

        unsafe {
            riscv::register::sstatus::set_spp(riscv::register::sstatus::SPP::Supervisor);
        }
        let sstatus_val: usize;
        unsafe {
            core::arch::asm!("csrr {}, sstatus", out(reg) sstatus_val);
        }
        ctx.guest_regs.sstatus = sstatus_val;
        ctx.guest_regs.sepc = VM_ENTRY;
    }
}

// ════════════════════════════════════════════════════════════════
//  AArch64  (Bootloader-style hypervisor — loads full ArceOS guest)
//
//  Since the ArceOS platform crate drops from EL2 to EL1 during
//  boot, we cannot use traditional EL2 virtualization with Stage-2
//  page tables. Instead, we use a bootloader approach:
//
//    1. Load the guest ArceOS binary from the FAT32 filesystem
//    2. Write it to a separate physical memory region (PA 0x44200000)
//    3. Set up an identity mapping for the trampoline page
//    4. Jump to the trampoline (identity-mapped VA = PA)
//    5. Trampoline disables MMU and jumps to guest at PA 0x44200000
//    6. Guest ArceOS boots independently with full hardware access
//
//  The guest runs u_6_0-style multitasking with CFS scheduler.
//  Timer, UART, and GIC are accessed directly by the guest.
//
//  Memory layout (QEMU 128MB RAM: 0x40000000 - 0x47FFFFFF):
//    - Hypervisor: PA 0x40000000 - 0x43FFFFFF (lower 64MB)
//    - Guest:      PA 0x44000000 - 0x47FFFFFF (upper 64MB)
// ════════════════════════════════════════════════════════════════

// Trampoline code for AArch64 guest handoff.
// Must be executed at an identity-mapped address (VA = PA).
// Disables MMU, caches, invalidates TLB/I-cache, then jumps to guest.
#[cfg(all(feature = "axstd", target_arch = "aarch64"))]
core::arch::global_asm!(
    ".section .text",
    ".balign 4096",
    ".global _aarch64_guest_trampoline",
    "_aarch64_guest_trampoline:",
    // x0 = guest entry physical address
    "mov x2, x0",                   // Save guest entry PA in x2
    // Disable MMU, D-cache, I-cache in SCTLR_EL1
    "mrs x1, sctlr_el1",
    "bic x1, x1, #(1 << 0)",        // M = 0: disable MMU
    "bic x1, x1, #(1 << 2)",        // C = 0: disable D-cache
    "bic x1, x1, #(1 << 12)",       // I = 0: disable I-cache
    "msr sctlr_el1, x1",
    "isb",
    // Invalidate TLB
    "tlbi vmalle1",
    "dsb ish",
    "isb",
    // Invalidate I-cache
    "ic iallu",
    "dsb ish",
    "isb",
    // Set x0 = 0 (no device tree; guest uses built-in defplat config)
    "mov x0, #0",
    // Jump to guest entry point (physical address, MMU is off)
    "br x2",
    ".global _aarch64_guest_trampoline_end",
    "_aarch64_guest_trampoline_end:",
);

#[cfg(all(feature = "axstd", target_arch = "aarch64"))]
fn aarch64_main() {
    use axhal::mem::{phys_to_virt, virt_to_phys, PhysAddr};
    use axhal::paging::MappingFlags;
    use memory_addr::{va, PAGE_SIZE_4K};

    ax_println!("Starting virtualization (bootloader mode)...");

    // Guest ArceOS binary is loaded at PA 0x44200000 (upper 64MB region).
    // This avoids overlap with the hypervisor kernel (at PA 0x40200000).
    const GUEST_KERNEL_PADDR: usize = VM_ENTRY; // 0x4420_0000

    // ── 1. Load guest binary from filesystem to physical memory ──
    let fname = "/sbin/gkernel";
    ax_println!("VM created success, loading images...");
    ax_println!("app: {}", fname);

    let ctx = axfs::ROOT_FS_CONTEXT.get().expect("Root FS not initialized");
    let file = axfs::File::open(ctx, fname).expect("Cannot open guest image");
    let mut total_bytes = 0usize;
    loop {
        let mut buf = [0u8; 4096];
        let n = axio::Read::read(&mut &file, &mut buf).expect("read");
        if n == 0 {
            break;
        }
        // Write directly to guest physical memory via hypervisor's linear mapping
        let dst_va = phys_to_virt(PhysAddr::from(GUEST_KERNEL_PADDR + total_bytes)).as_usize();
        unsafe {
            core::ptr::copy_nonoverlapping(
                buf.as_ptr(),
                dst_va as *mut u8,
                n,
            );
        }
        total_bytes += n;
        if n < 4096 {
            break;
        }
    }
    ax_println!("Loaded {} bytes to PA {:#x}", total_bytes, GUEST_KERNEL_PADDR);

    // ── 2. Clean D-cache for guest binary ──
    // Ensures data is written to main memory before MMU & caches are disabled.
    let guest_va_base = phys_to_virt(PhysAddr::from(GUEST_KERNEL_PADDR)).as_usize();
    unsafe {
        let mut off = 0usize;
        while off < total_bytes {
            core::arch::asm!("dc cvau, {}", in(reg) (guest_va_base + off));
            off += 64; // cache line size
        }
        core::arch::asm!("dsb ish");
        core::arch::asm!("ic iallu");
        core::arch::asm!("dsb ish");
        core::arch::asm!("isb");
    }

    // ── 3. Get physical address of the trampoline ──
    extern "C" {
        fn _aarch64_guest_trampoline();
    }
    let trampoline_va = _aarch64_guest_trampoline as usize;
    let trampoline_pa = usize::from(virt_to_phys(trampoline_va.into()));
    let trampoline_page_pa = trampoline_pa & !0xFFF;
    ax_println!(
        "Trampoline at VA {:#x} -> PA {:#x} (page {:#x})",
        trampoline_va, trampoline_pa, trampoline_page_pa
    );

    // ── 4. Create identity mapping for trampoline page in TTBR0 ──
    // The trampoline must be at an identity-mapped address (VA = PA) so
    // that it can disable MMU without the instruction stream becoming invalid.
    let flags = MappingFlags::READ | MappingFlags::WRITE
        | MappingFlags::EXECUTE | MappingFlags::USER;

    let mut identity = axmm::AddrSpace::new_empty(va!(0x0), 0x4800_0000).unwrap();
    identity.map_linear(
        trampoline_page_pa.into(),
        PhysAddr::from(trampoline_page_pa),
        PAGE_SIZE_4K,
        flags,
    ).expect("identity-map trampoline page");

    // ── 5. Switch TTBR0 to identity page table ──
    let pt_root = identity.page_table_root();
    unsafe {
        core::arch::asm!(
            "msr ttbr0_el1, {val}",
            "isb",
            "tlbi vmalle1is",
            "dsb ish",
            "isb",
            val = in(reg) usize::from(pt_root) as u64,
        );
    }

    // Prevent the identity AddrSpace from being dropped (which would free
    // the page table pages). We are about to jump and never return.
    core::mem::forget(identity);

    // ── 6. Disable interrupts and jump to identity-mapped trampoline ──
    // The trampoline will:
    //   a) Disable MMU, D-cache, I-cache
    //   b) Invalidate TLB and I-cache
    //   c) Jump to guest at PA 0x44200000
    // The guest ArceOS boots at EL1 with MMU off, just like a normal boot.
    ax_println!(
        "Entering guest at PA {:#x} via trampoline at PA {:#x}...",
        GUEST_KERNEL_PADDR,
        trampoline_pa
    );
    unsafe {
        core::arch::asm!(
            "msr daifset, #0xf",       // Mask all exceptions (DAIF)
            "mov x0, {entry}",         // x0 = guest entry physical address
            "br {tramp}",              // Jump to identity-mapped trampoline
            entry = in(reg) GUEST_KERNEL_PADDR as u64,
            tramp = in(reg) trampoline_pa as u64,
            options(noreturn),
        );
    }
}

// ════════════════════════════════════════════════════════════════
//  x86_64  (AMD SVM hypervisor — long-mode guest with NPT)
//
//  The guest runs in 64-bit long mode inside an SVM container.
//  The hypervisor creates initial page tables, GDT, and VMCB for
//  the guest, then uses VMRUN to execute it.
//
//  Nested Page Tables (NPT) provide GPA→HPA translation.
//  Guest page tables provide GVA→GPA translation.
//
//  VMMCALL hypercalls are used for console I/O and shutdown.
//  NPF (Nested Page Fault) is used for pflash emulation.
// ════════════════════════════════════════════════════════════════

#[cfg(all(feature = "axstd", target_arch = "x86_64"))]
fn x86_64_main() {
    use alloc::boxed::Box;
    use alloc::sync::Arc;
    use x86_64_svm::vmcb::*;
    use x86_64_svm::svm::*;
    use memory_addr::va;
    use axhal::paging::{MappingFlags, PageSize};
    use axmm::backend::{Backend, SharedPages};
    use memory_addr::PAGE_SIZE_4K;

    ax_println!("Starting virtualization...");

    // ── 1. Check AMD SVM support ──
    let (_, _, ecx, _) = unsafe { cpuid(0x8000_0001) };
    if ecx & (1 << 2) == 0 {
        panic!("CPU does not support AMD SVM!");
    }

    // ── 2. Enable SVM ──
    unsafe {
        let efer = rdmsr(MSR_EFER);
        wrmsr(MSR_EFER, efer | EFER_SVME);
    }

    // ── 3. Allocate host-save area ──
    #[repr(C, align(4096))]
    struct Page4K([u8; 4096]);
    let host_save = Box::new(Page4K([0u8; 4096]));
    let host_save_pa = virt_to_phys_ptr(&host_save.0[0]);
    unsafe {
        wrmsr(MSR_VM_HSAVE_PA, host_save_pa);
    }

    let host_vmcb = Box::new(Page4K([0u8; 4096]));
    let host_vmcb_pa = virt_to_phys_ptr(&host_vmcb.0[0]);

    // ── 4. Allocate IOPM and MSRPM ──
    #[repr(C, align(4096))]
    struct Iopm([u8; 12288]);
    #[repr(C, align(4096))]
    struct Msrpm([u8; 8192]);
    let iopm = Box::new(Iopm([0u8; 12288]));
    let msrpm = Box::new(Msrpm([0u8; 8192]));
    let iopm_pa = virt_to_phys_ptr(&iopm.0[0]);
    let msrpm_pa = virt_to_phys_ptr(&msrpm.0[0]);

    // ── 5. Create NPT and pre-allocate guest RAM ──
    let mut npt = axmm::AddrSpace::new_empty(va!(0x0), 0x1_0000_0000).unwrap();

    let flags = MappingFlags::READ | MappingFlags::WRITE
        | MappingFlags::EXECUTE | MappingFlags::USER;

    const GUEST_RAM_SIZE: usize = 0x20_0000; // 2MB
    ax_println!("Pre-allocating {} KB guest RAM at GPA 0x0...", GUEST_RAM_SIZE / 1024);
    let ram_pages = Arc::new(
        SharedPages::new(GUEST_RAM_SIZE, PageSize::Size4K)
            .expect("alloc guest RAM"),
    );
    npt.map(
        0x0usize.into(),
        GUEST_RAM_SIZE,
        flags,
        true,
        Backend::new_shared(0x0usize.into(), ram_pages),
    ).expect("map guest RAM");

    // ── 6. Write guest page tables into NPT-mapped memory ──
    const PTE_PRESENT: u64 = 1;
    const PTE_RW: u64 = 1 << 1;
    const PTE_USER: u64 = 1 << 2;
    const PTE_PS: u64 = 1 << 7;
    const PT_FLAGS: u64 = PTE_PRESENT | PTE_RW | PTE_USER;

    npt.write(0x1000usize.into(), &(0x2000u64 | PT_FLAGS).to_le_bytes())
        .expect("write PML4");
    npt.write(0x2000usize.into(), &(0x3000u64 | PT_FLAGS).to_le_bytes())
        .expect("write PDPT[0]");
    npt.write((0x2000 + 3 * 8usize).into(), &(0x4000u64 | PT_FLAGS).to_le_bytes())
        .expect("write PDPT[3]");
    npt.write(0x3000usize.into(), &(0x0u64 | PT_FLAGS | PTE_PS).to_le_bytes())
        .expect("write PD0[0]");
    npt.write((0x4000 + 510 * 8usize).into(), &(0xFFC0_0000u64 | PT_FLAGS | PTE_PS).to_le_bytes())
        .expect("write PD3[510]");

    // ── 7. Write GDT into guest memory (GPA 0x5000) ──
    let gdt: [u64; 4] = [
        0x0000_0000_0000_0000, // null
        0x00CF_9B00_0000_FFFF, // 32-bit code
        0x00AF_9B00_0000_FFFF, // 64-bit code
        0x00CF_9300_0000_FFFF, // data
    ];
    for (i, &entry) in gdt.iter().enumerate() {
        npt.write((0x5000 + i * 8).into(), &entry.to_le_bytes())
            .expect("write GDT");
    }

    // ── 8. Load guest binary at GPA VM_ENTRY (0x10000) ──
    {
        let fname = "/sbin/gkernel";
        ax_println!("VM created success, loading images...");
        ax_println!("app: {}", fname);
        let ctx = axfs::ROOT_FS_CONTEXT.get().expect("Root FS not initialized");
        let file = axfs::File::open(ctx, fname).expect("Cannot open guest image");
        let mut offset = 0usize;
        let mut total_bytes = 0usize;
        loop {
            let mut buf = [0u8; 4096];
            let n = axio::Read::read(&mut &file, &mut buf).expect("read");
            if n == 0 {
                break;
            }
            total_bytes += n;
            npt.write((VM_ENTRY + offset).into(), &buf[..n])
                .expect("write guest binary");
            offset += n;
            if n < 4096 {
                break;
            }
        }
        ax_println!("Loaded {} bytes from {}", total_bytes, fname);
    }

    let npt_root_pa: u64 = usize::from(npt.page_table_root()) as u64;

    // ── 9. Build VMCB for 64-bit long mode ──
    let mut vmcb = Box::new(Vmcb::new());

    vmcb.write_u32(CTRL_INTERCEPT_MISC2, INTERCEPT_VMRUN | INTERCEPT_VMMCALL);
    vmcb.write_u64(CTRL_IOPM_BASE, iopm_pa);
    vmcb.write_u64(CTRL_MSRPM_BASE, msrpm_pa);
    vmcb.write_u32(CTRL_GUEST_ASID, 1);
    vmcb.write_u64(CTRL_NP_ENABLE, 1);
    vmcb.write_u64(CTRL_NCR3, npt_root_pa);

    vmcb.set_segment(SAVE_CS, 0x10, 0x0A9B, 0xFFFF_FFFF, 0);
    vmcb.set_segment(SAVE_DS, 0x18, 0x0C93, 0xFFFF_FFFF, 0);
    vmcb.set_segment(SAVE_ES, 0x18, 0x0C93, 0xFFFF_FFFF, 0);
    vmcb.set_segment(SAVE_SS, 0x18, 0x0C93, 0xFFFF_FFFF, 0);
    vmcb.set_segment(SAVE_FS, 0, 0, 0, 0);
    vmcb.set_segment(SAVE_GS, 0, 0, 0, 0);
    vmcb.set_segment(SAVE_GDTR, 0, 0, 31, 0x5000);
    vmcb.set_segment(SAVE_IDTR, 0, 0, 0xFFF, 0);
    vmcb.set_segment(SAVE_TR, 0, 0x008B, 0x67, 0);
    vmcb.set_segment(SAVE_LDTR, 0, 0x0082, 0, 0);

    vmcb.write_u64(SAVE_CR0, 0x8001_0011);
    vmcb.write_u64(SAVE_CR3, 0x1000);
    vmcb.write_u64(SAVE_CR4, 0x00A0);
    vmcb.write_u64(SAVE_EFER, EFER_SVME | (1 << 8) | (1 << 10) | (1 << 11));

    vmcb.write_u64(SAVE_DR6, 0xFFFF_0FF0);
    vmcb.write_u64(SAVE_DR7, 0x0400);
    vmcb.write_u64(SAVE_RFLAGS, 0x2);
    vmcb.write_u64(SAVE_RIP, VM_ENTRY as u64);
    vmcb.write_u64(SAVE_RSP, 0x80000);

    let vmcb_pa = virt_to_phys_ptr(&vmcb.data[0]);

    // ── 10. Create guest GPR save area ──
    let mut gprs = SvmGuestGprs::new();

    // ── 11. Run guest in loop ──
    ax_println!("Entering VM run loop...");
    loop {
        unsafe {
            _run_guest(vmcb_pa, host_vmcb_pa, &mut gprs);
        }

        let exit_code = vmcb.exit_code();

        match exit_code {
            VMEXIT_VMMCALL => {
                let guest_rax = vmcb.guest_rax();
                let func = guest_rax & 0xFF;

                if guest_rax == 0x84000008 {
                    ax_println!("Shutdown vm normally!");
                    break;
                } else if func == 1 {
                    let ch = ((guest_rax >> 8) & 0xFF) as u8;
                    ax_print!("{}", ch as char);
                    let rip = vmcb.guest_rip();
                    vmcb.write_u64(SAVE_RIP, rip + 3);
                } else {
                    let rip = vmcb.guest_rip();
                    vmcb.write_u64(SAVE_RIP, rip + 3);
                }
            }
            VMEXIT_NPF => {
                let fault_addr = vmcb.exit_info2();
                let page_addr = (fault_addr & !0xFFF) as usize;

                let is_pflash = page_addr >= 0xFFC0_0000 && page_addr < 0x1_0000_0000;

                let pages = Arc::new(
                    SharedPages::new(PAGE_SIZE_4K, PageSize::Size4K)
                        .expect("alloc page for NPF"),
                );
                npt.map(
                    page_addr.into(),
                    PAGE_SIZE_4K,
                    flags,
                    true,
                    Backend::new_shared(page_addr.into(), pages),
                ).expect("map NPF page");

                if is_pflash {
                    npt.write(page_addr.into(), &0x646c6670u32.to_le_bytes())
                        .expect("write pflash magic");
                }
            }
            _ => {
                ax_println!(
                    "Unexpected VMEXIT: exit_code={:#x}, info1={:#x}, info2={:#x}, RIP={:#x}",
                    exit_code,
                    vmcb.exit_info1(),
                    vmcb.exit_info2(),
                    vmcb.guest_rip(),
                );
                break;
            }
        }
    }

    ax_println!("Hypervisor ok!");

    unsafe {
        core::arch::asm!(
            "mov dx, 0x604",
            "mov ax, 0x2000",
            "out dx, ax",
        );
    }
    panic!("Hypervisor ok!");

    fn virt_to_phys_ptr(p: *const u8) -> u64 {
        use axhal::mem::virt_to_phys;
        let va = memory_addr::VirtAddr::from(p as usize);
        usize::from(virt_to_phys(va)) as u64
    }
}
