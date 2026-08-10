//! Live-device corroboration for the success-only device-object counters (issue #88).
//!
//! # What this is, and what it is deliberately NOT
//!
//! The rule under test elsewhere is *a failed Vulkan call increments no counter*. That rule is
//! settled by the host-free seam tests in `rust/src/counters.rs`, which drive both polarities of
//! `counters::device_objects` with no driver at all. **Those tests are the authority.**
//!
//! This binary asks a smaller, purely empirical question that no host-free test can answer:
//! *on this machine, is the refusable path the seam's `Failed` polarity is written for actually
//! reachable?* The EP creates its descriptor pool with `maxSets(1)`
//! (`vk/pipeline.rs::DispatchDescriptorPool::new`), and asking such a pool for a second set is
//! **not** a spec-guaranteed failure — a conformant driver may simply hand one over. So the
//! honest outcomes are three, and two of them are not passes:
//!
//! * `CORROBORATED` — the driver refused, and wrote no handle.
//! * `INCONCLUSIVE` — the driver was lenient. Conformant, and the question stays open.
//! * `SKIP` — no Vulkan 1.1 compute device was reachable. **Nothing was observed.**
//!
//! Only a fourth outcome, `CONTRADICTED` (an error return that still produced a handle), exits
//! non-zero, because only that one disagrees with the premise the counters rest on.
//!
//! # Why `harness = false`
//!
//! A libtest `#[test]` that passes has its stdout and stderr **captured and discarded**. A
//! corroboration whose SKIP and INCONCLUSIVE verdicts are only visible under `--nocapture` is a
//! corroboration nobody reads: on every machine without the right driver it would print a green
//! `ok` and say nothing. `harness = false` makes this a plain `main()`, so its verdict is on the
//! terminal in the default hosted command (`cargo test --manifest-path rust/Cargo.toml`) with no
//! extra flag. `rust/tests/corroboration_report.rs` asserts that declaration is still in
//! `Cargo.toml`.
//!
//! # Portability
//!
//! Instance and device are requested at **Vulkan 1.1** and nothing above it is used, matching the
//! EP's floor (`docs/DESIGN.md`). No performance claim is made or implied anywhere in here.

#[path = "corroboration/mod.rs"]
mod corroboration;

use ash::vk;
use corroboration::{Exhaustion, Report, classify};

/// Opt out without editing code — for hosts where creating a device is undesirable.
const ENV_SKIP: &str = "ONNXRUNTIME_EP_VULKAN_SKIP_DEVICE_CORROBORATION";

fn main() {
    let report = run();
    // Straight to stdout, unconditionally, in every outcome. This is the whole reason the target
    // is `harness = false`.
    print!("{}", report.render());
    let code = report.exit_code();
    if code != 0 {
        eprintln!(
            "device-object counter corroboration FAILED: {}",
            report.token()
        );
    }
    std::process::exit(code);
}

fn skip(reason: impl Into<String>) -> Report {
    Report::Skip {
        reason: reason.into(),
    }
}

fn run() -> Report {
    if std::env::var_os(ENV_SKIP).is_some() {
        return skip(format!("{ENV_SKIP} is set"));
    }

    // SAFETY: `Entry::load` dynamically loads the Vulkan loader. It is unsafe because the loader
    // is arbitrary native code; there is no alternative and the EP itself does exactly this.
    let entry = match unsafe { ash::Entry::load() } {
        Ok(e) => e,
        Err(e) => return skip(format!("no Vulkan loader on this host ({e})")),
    };

    let app_info = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_1);
    let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);
    // SAFETY: `create_info` is valid and outlives the call; no extensions or layers are enabled.
    let instance = match unsafe { entry.create_instance(&create_info, None) } {
        Ok(i) => i,
        Err(e) => return skip(format!("vkCreateInstance at Vulkan 1.1 failed ({e})")),
    };

    let report = probe(&instance);
    // SAFETY: `instance` is live, was created here, and every child created below is destroyed
    // inside `probe` before it returns.
    unsafe { instance.destroy_instance(None) };
    report
}

fn probe(instance: &ash::Instance) -> Report {
    // SAFETY: `instance` is live.
    let physical = match unsafe { instance.enumerate_physical_devices() } {
        Ok(d) if !d.is_empty() => d,
        Ok(_) => return skip("the loader reported zero physical devices"),
        Err(e) => return skip(format!("vkEnumeratePhysicalDevices failed ({e})")),
    };

    for pd in physical {
        // SAFETY: `pd` came from this instance.
        let props = unsafe { instance.get_physical_device_properties(pd) };
        if vk::api_version_major(props.api_version) < 1
            || (vk::api_version_major(props.api_version) == 1
                && vk::api_version_minor(props.api_version) < 1)
        {
            continue;
        }
        let name = device_name(&props);
        // SAFETY: `pd` came from this instance.
        let families = unsafe { instance.get_physical_device_queue_family_properties(pd) };
        let Some(family) = families
            .iter()
            .position(|f| f.queue_flags.contains(vk::QueueFlags::COMPUTE))
        else {
            continue;
        };
        return probe_device(instance, pd, family as u32, &name);
    }
    skip("no physical device advertised Vulkan 1.1 with a compute queue family")
}

fn device_name(props: &vk::PhysicalDeviceProperties) -> String {
    let bytes: Vec<u8> = props
        .device_name
        .iter()
        .take_while(|&&c| c != 0)
        .map(|&c| c as u8)
        .collect();
    String::from_utf8_lossy(&bytes).to_string()
}

fn probe_device(
    instance: &ash::Instance,
    pd: vk::PhysicalDevice,
    family: u32,
    name: &str,
) -> Report {
    let priorities = [1.0f32];
    let queue_info = [vk::DeviceQueueCreateInfo::default()
        .queue_family_index(family)
        .queue_priorities(&priorities)];
    let device_info = vk::DeviceCreateInfo::default().queue_create_infos(&queue_info);
    // SAFETY: `pd` came from `instance`; `device_info` is valid and outlives the call.
    let device = match unsafe { instance.create_device(pd, &device_info, None) } {
        Ok(d) => d,
        Err(e) => return skip(format!("vkCreateDevice on `{name}` failed ({e})")),
    };

    let report = probe_pool(&device, name);
    // SAFETY: every child object created in `probe_pool` is destroyed before it returns, and no
    // work was ever submitted to this device.
    unsafe { device.destroy_device(None) };
    report
}

fn probe_pool(device: &ash::Device, name: &str) -> Report {
    let bindings = [vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)];
    let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    // SAFETY: `device` is live; `layout_info` is valid and outlives the call.
    let layout = match unsafe { device.create_descriptor_set_layout(&layout_info, None) } {
        Ok(l) => l,
        Err(e) => {
            return skip(format!(
                "vkCreateDescriptorSetLayout on `{name}` failed ({e})"
            ));
        }
    };

    // `maxSets(1)` mirrors production. The descriptor budget is deliberately generous (8 storage
    // buffer descriptors for a 1-descriptor layout) so that a refusal, if it comes, can only be
    // about the SET limit — the thing production's pool actually constrains — and not about
    // running out of descriptors, which would be a different question wearing the same answer.
    let pool_sizes = [vk::DescriptorPoolSize {
        ty: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count: 8,
    }];
    let pool_info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(1)
        .pool_sizes(&pool_sizes);
    // SAFETY: `device` is live; `pool_info` is valid and outlives the call.
    let pool = match unsafe { device.create_descriptor_pool(&pool_info, None) } {
        Ok(p) => p,
        Err(e) => {
            // SAFETY: `layout` was created above and nothing references it.
            unsafe { device.destroy_descriptor_set_layout(layout, None) };
            return skip(format!("vkCreateDescriptorPool on `{name}` failed ({e})"));
        }
    };

    let report = observe_exhaustion(device, pool, layout, name);

    // SAFETY: no queue work was ever submitted, so no descriptor set from this pool is in use.
    unsafe {
        device.destroy_descriptor_pool(pool, None);
        device.destroy_descriptor_set_layout(layout, None);
    }
    report
}

fn observe_exhaustion(
    device: &ash::Device,
    pool: vk::DescriptorPool,
    layout: vk::DescriptorSetLayout,
    name: &str,
) -> Report {
    let layouts = [layout];
    let alloc_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(pool)
        .set_layouts(&layouts);

    // The first set must succeed. If it does not, the premise of the probe (that this pool can
    // serve exactly one set) is not established and there is nothing to over-allocate past.
    // SAFETY: `pool` and `layout` are live and were created from `device`.
    let first = unsafe { device.allocate_descriptor_sets(&alloc_info) };
    match first {
        Ok(sets) if sets.iter().all(|s| *s != vk::DescriptorSet::null()) => {}
        Ok(_) => {
            return Report::Contradicted {
                device: name.to_string(),
                detail: "vkAllocateDescriptorSets reported success and wrote a NULL \
                         VkDescriptorSet handle"
                    .to_string(),
            };
        }
        Err(e) => {
            return skip(format!(
                "the FIRST descriptor set could not be allocated on `{name}` ({e}), so there is \
                 no exhaustion boundary to probe"
            ));
        }
    }

    // The second set is the question. A conformant driver may refuse it or serve it.
    // SAFETY: as above.
    let second = unsafe { device.allocate_descriptor_sets(&alloc_info) };
    match second {
        Ok(sets) => {
            let _ = sets;
            classify(name, Exhaustion::Allowed, true)
        }
        Err(_) => {
            // ash returns `Err` without handing back the output slice, so no handle can have
            // reached this frame. That is the observation the classifier needs: an error return
            // that produced no object.
            classify(name, Exhaustion::Refused, false)
        }
    }
}
