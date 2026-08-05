//! A deliberate, controllable VRAM co-tenant.
//!
//! WHY THIS EXISTS
//! ---------------
//! The ctx-4096 device loss in the device-resident lane reproduces, then hides on re-run.
//! Across two of my gate arms the losses were **time-ordered**, not treatment-ordered
//! (L,L,L,C,C then C,C,C), which is the signature of a *box state* variable rather than of
//! anything the EP itself did differently. The EP's own peak device footprint is constant to
//! the byte across losing and clean runs (`alloc_high_water_bytes = 5_518_426_760` in every
//! capture), so the variable is not ours -- it is how much of the 8188 MiB board somebody
//! else was holding at the time.
//!
//! A hypothesis about a variable I cannot set is not testable. This binary lets me set it:
//! it holds a stated number of MiB of DEVICE_LOCAL memory on a stated device, and holds it
//! until it is told to stop. That converts "the fault hides on re-run" into "the fault is
//! present when the board is full and absent when it is not" -- or falsifies that outright.
//!
//! It is an *instrument*, not part of the EP. It deliberately does not link the EP, does not
//! read any EP environment variable, and does not use gpu-allocator: if it shared machinery
//! with the thing under test, a shared defect would cancel out of the comparison.
//!
//! USAGE
//!   cargo run --release --example vram_occupant -- --mib 3000 --seconds 1800
//!   cargo run --release --example vram_occupant -- --mib 3000 --until-deleted <path>
//!   cargo run --release --example vram_occupant -- --list
//!
//! Exit codes:  0 held and released cleanly | 2 usage | 3 could not hold the stated amount

use ash::vk;
use std::ffi::CStr;
use std::time::{Duration, Instant};

const CHUNK_MIB: u64 = 256;

fn device_name(props: &vk::PhysicalDeviceProperties) -> String {
    unsafe { CStr::from_ptr(props.device_name.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

struct Args {
    mib: u64,
    seconds: u64,
    until_deleted: Option<String>,
    device_index: Option<usize>,
    list: bool,
    budget: bool,
}

/// Report `VK_EXT_memory_budget`'s view of every DEVICE_LOCAL heap.
///
/// This exists because `nvidia-smi` **cannot see this board**: with 2048 MiB held and touched
/// from a queue, `--query-gpu=memory.used` still reported `0 MiB` and `--query-compute-apps`
/// reported `[N/A]` for per-process memory. On this laptop NVIDIA part, NVML is not an
/// occupancy instrument. `heapBudget` is -- and it is the same number the EP would have to
/// consult to know it is about to overcommit, so measuring with it here is measuring with the
/// instrument the fix would install, not with a proxy for it.
fn report_budget(instance: &ash::Instance, pd: vk::PhysicalDevice, when: &str) {
    let supported = match unsafe { instance.enumerate_device_extension_properties(pd) } {
        Ok(exts) => exts.iter().any(|e| {
            unsafe { CStr::from_ptr(e.extension_name.as_ptr()) }.to_string_lossy()
                == "VK_EXT_memory_budget"
        }),
        Err(_) => false,
    };
    if !supported {
        println!(
            "budget[{when}]: VK_EXT_memory_budget NOT SUPPORTED - no headroom readout exists here"
        );
        return;
    }
    let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
    let mut props2 = vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
    unsafe { instance.get_physical_device_memory_properties2(pd, &mut props2) };
    let mem = props2.memory_properties;
    for h in 0..mem.memory_heap_count as usize {
        if !mem.memory_heaps[h]
            .flags
            .contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
        {
            continue;
        }
        let mib = |b: u64| b / (1024 * 1024);
        println!(
            "budget[{when}]: heap {h} size {} MiB  budget {} MiB  usage {} MiB  headroom {} MiB",
            mib(mem.memory_heaps[h].size),
            mib(budget.heap_budget[h]),
            mib(budget.heap_usage[h]),
            mib(budget.heap_budget[h].saturating_sub(budget.heap_usage[h])),
        );
    }
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        mib: 0,
        seconds: 0,
        until_deleted: None,
        device_index: None,
        list: false,
        budget: false,
    };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let need = |i: usize| -> Result<String, String> {
            argv.get(i + 1)
                .cloned()
                .ok_or_else(|| format!("{} needs a value", argv[i]))
        };
        match argv[i].as_str() {
            "--mib" => {
                a.mib = need(i)?.parse().map_err(|_| "--mib must be an integer")?;
                i += 2;
            }
            "--seconds" => {
                a.seconds = need(i)?
                    .parse()
                    .map_err(|_| "--seconds must be an integer")?;
                i += 2;
            }
            "--until-deleted" => {
                a.until_deleted = Some(need(i)?);
                i += 2;
            }
            "--device-index" => {
                a.device_index = Some(
                    need(i)?
                        .parse()
                        .map_err(|_| "--device-index must be an integer")?,
                );
                i += 2;
            }
            "--list" => {
                a.list = true;
                i += 1;
            }
            "--budget" => {
                a.budget = true;
                i += 1;
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    let probe_only = a.budget && a.mib == 0;
    if !a.list && !probe_only && a.mib == 0 {
        return Err("--mib is required (or --list, or --budget alone to probe)".to_string());
    }
    if !a.list && !probe_only && a.seconds == 0 && a.until_deleted.is_none() {
        return Err("one of --seconds or --until-deleted is required".to_string());
    }
    Ok(a)
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("vram_occupant: {e}");
            eprintln!(
                "USAGE: --mib N [--seconds N | --until-deleted PATH] [--device-index N] | --list"
            );
            std::process::exit(2);
        }
    };

    let entry = match unsafe { ash::Entry::load() } {
        Ok(e) => e,
        Err(e) => {
            eprintln!("vram_occupant: could not load the Vulkan loader: {e}");
            std::process::exit(3);
        }
    };

    let app = vk::ApplicationInfo::default().api_version(vk::make_api_version(0, 1, 1, 0));
    let ici = vk::InstanceCreateInfo::default().application_info(&app);
    let instance = match unsafe { entry.create_instance(&ici, None) } {
        Ok(i) => i,
        Err(e) => {
            eprintln!("vram_occupant: vkCreateInstance failed: {e}");
            std::process::exit(3);
        }
    };

    let pds = match unsafe { instance.enumerate_physical_devices() } {
        Ok(p) => p,
        Err(e) => {
            eprintln!("vram_occupant: vkEnumeratePhysicalDevices failed: {e}");
            std::process::exit(3);
        }
    };

    if args.list {
        for (idx, pd) in pds.iter().enumerate() {
            let props = unsafe { instance.get_physical_device_properties(*pd) };
            let mem = unsafe { instance.get_physical_device_memory_properties(*pd) };
            let local: u64 = (0..mem.memory_heap_count as usize)
                .filter(|h| {
                    mem.memory_heaps[*h]
                        .flags
                        .contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
                })
                .map(|h| mem.memory_heaps[h].size)
                .max()
                .unwrap_or(0);
            println!(
                "device {idx}: {} - largest DEVICE_LOCAL heap {} MiB",
                device_name(&props),
                local / (1024 * 1024)
            );
        }
        unsafe { instance.destroy_instance(None) };
        return;
    }

    let chosen = match args.device_index {
        Some(i) if i < pds.len() => pds[i],
        Some(i) => {
            eprintln!(
                "vram_occupant: --device-index {i} but only {} device(s)",
                pds.len()
            );
            std::process::exit(2);
        }
        None => match pds.first() {
            Some(p) => *p,
            None => {
                eprintln!("vram_occupant: no Vulkan physical devices");
                std::process::exit(3);
            }
        },
    };

    let props = unsafe { instance.get_physical_device_properties(chosen) };
    let mem_props = unsafe { instance.get_physical_device_memory_properties(chosen) };

    if args.budget {
        println!("vram_occupant: device {}", device_name(&props));
        report_budget(&instance, chosen, "before");
        if args.mib == 0 {
            unsafe { instance.destroy_instance(None) };
            return;
        }
    }

    // A DEVICE_LOCAL type that is NOT HOST_VISIBLE: on a discrete board that is the one that
    // actually costs VRAM. Accepting any DEVICE_LOCAL type would let the driver satisfy us out
    // of a BAR window and the co-tenancy would be fictional.
    let mut type_index: Option<u32> = None;
    for t in 0..mem_props.memory_type_count as usize {
        let mt = mem_props.memory_types[t];
        let heap = mem_props.memory_heaps[mt.heap_index as usize];
        if mt
            .property_flags
            .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
            && !mt
                .property_flags
                .contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
            && heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
        {
            type_index = Some(t as u32);
            break;
        }
    }
    let Some(type_index) = type_index else {
        eprintln!(
            "vram_occupant: {} exposes no DEVICE_LOCAL-and-not-HOST_VISIBLE memory type; \
             holding memory here would not be board occupancy and I will not pretend it is",
            device_name(&props)
        );
        unsafe { instance.destroy_instance(None) };
        std::process::exit(3);
    };

    let qfams = unsafe { instance.get_physical_device_queue_family_properties(chosen) };
    let qfi = qfams
        .iter()
        .position(|q| q.queue_flags.contains(vk::QueueFlags::TRANSFER))
        .unwrap_or(0) as u32;

    let prios = [1.0f32];
    let qci = [vk::DeviceQueueCreateInfo::default()
        .queue_family_index(qfi)
        .queue_priorities(&prios)];
    let dci = vk::DeviceCreateInfo::default().queue_create_infos(&qci);
    let device = match unsafe { instance.create_device(chosen, &dci, None) } {
        Ok(d) => d,
        Err(e) => {
            eprintln!("vram_occupant: vkCreateDevice failed: {e}");
            std::process::exit(3);
        }
    };
    let queue = unsafe { device.get_device_queue(qfi, 0) };

    let pool = unsafe {
        device.create_command_pool(
            &vk::CommandPoolCreateInfo::default().queue_family_index(qfi),
            None,
        )
    }
    .expect("vkCreateCommandPool");

    let chunk = CHUNK_MIB * 1024 * 1024;
    let want_chunks = args.mib.div_ceil(CHUNK_MIB);
    let mut held: Vec<(vk::DeviceMemory, vk::Buffer)> = Vec::new();

    println!(
        "vram_occupant: holding {} MiB on device {} (memory type {type_index}) in {CHUNK_MIB} MiB chunks",
        want_chunks * CHUNK_MIB,
        device_name(&props),
    );

    for n in 0..want_chunks {
        let buf = match unsafe {
            device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(chunk)
                    .usage(vk::BufferUsageFlags::TRANSFER_DST)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )
        } {
            Ok(b) => b,
            Err(e) => {
                eprintln!("vram_occupant: vkCreateBuffer failed at chunk {n}: {e}");
                break;
            }
        };
        let mai = vk::MemoryAllocateInfo::default()
            .allocation_size(chunk)
            .memory_type_index(type_index);
        let mem = match unsafe { device.allocate_memory(&mai, None) } {
            Ok(m) => m,
            Err(e) => {
                eprintln!("vram_occupant: vkAllocateMemory failed at chunk {n}: {e}");
                unsafe { device.destroy_buffer(buf, None) };
                break;
            }
        };
        if let Err(e) = unsafe { device.bind_buffer_memory(buf, mem, 0) } {
            eprintln!("vram_occupant: vkBindBufferMemory failed at chunk {n}: {e}");
            unsafe {
                device.free_memory(mem, None);
                device.destroy_buffer(buf, None);
            }
            break;
        }
        held.push((mem, buf));
    }

    // Allocation alone does not prove residency: on WDDM an allocation can be created and left
    // unbacked until first use. Touching every chunk from the queue forces the driver to make
    // them resident at least once, which is what a real co-tenant does.
    if !held.is_empty() {
        let cbs = unsafe {
            device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        }
        .expect("vkAllocateCommandBuffers");
        let cb = cbs[0];
        unsafe {
            device
                .begin_command_buffer(
                    cb,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .expect("vkBeginCommandBuffer");
            for (_, buf) in &held {
                device.cmd_fill_buffer(cb, *buf, 0, vk::WHOLE_SIZE, 0xA5A5_A5A5);
            }
            device.end_command_buffer(cb).expect("vkEndCommandBuffer");
            let cbs_arr = [cb];
            let submit = vk::SubmitInfo::default().command_buffers(&cbs_arr);
            device
                .queue_submit(queue, &[submit], vk::Fence::null())
                .expect("vkQueueSubmit");
            device.queue_wait_idle(queue).expect("vkQueueWaitIdle");
        }
    }

    let got = held.len() as u64 * CHUNK_MIB;
    println!(
        "vram_occupant: HELD {got} MiB of {} MiB requested, touched from the queue",
        want_chunks * CHUNK_MIB
    );
    if args.budget {
        report_budget(&instance, chosen, "holding");
    }
    if got < want_chunks * CHUNK_MIB {
        eprintln!(
            "vram_occupant: could not hold the stated amount - report {got} MiB, not the request"
        );
    }

    let start = Instant::now();
    loop {
        if let Some(path) = &args.until_deleted {
            if !std::path::Path::new(path).exists() {
                println!("vram_occupant: sentinel {path} gone - releasing");
                break;
            }
        }
        if args.seconds > 0 && start.elapsed() >= Duration::from_secs(args.seconds) {
            println!("vram_occupant: {} s elapsed - releasing", args.seconds);
            break;
        }
        std::thread::sleep(Duration::from_millis(500));
    }

    unsafe {
        for (mem, buf) in held {
            device.destroy_buffer(buf, None);
            device.free_memory(mem, None);
        }
        device.destroy_command_pool(pool, None);
        device.destroy_device(None);
        instance.destroy_instance(None);
    }
    println!(
        "vram_occupant: released after {:.1} s",
        start.elapsed().as_secs_f64()
    );
    if got < want_chunks * CHUNK_MIB {
        std::process::exit(3);
    }
}
