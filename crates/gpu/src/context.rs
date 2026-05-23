//! WGPU device + queue context.  Initialized once per process and
//! cached.

use std::sync::OnceLock;

use wgpu::{Adapter, Device, Instance, Queue, RequestAdapterOptions};

/// Lazy GPU context — initialized on first call to [`GpuContext::get`].
/// All fields are cheap to clone (they're internally Arc'd by wgpu).
pub struct GpuContext {
    pub instance: Instance,
    pub adapter: Adapter,
    pub device: Device,
    pub queue: Queue,
}

impl GpuContext {
    /// Initialise wgpu (blocks on adapter selection and device
    /// creation).  On Mac this picks Metal; on Linux Vulkan; etc.
    pub fn new() -> Result<Self, GpuInitError> {
        let instance = Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok_or_else(|| GpuInitError::NoAdapter("no compatible adapter found".to_string()))?;
        // The default `wgpu::Limits` cap `max_storage_buffers_per_shader_stage`
        // at 8, which the Verlet kernel exceeds (11 storage buffers).
        // Apple Silicon supports 128+; bump to whatever the adapter
        // actually offers so we don't have to artificially pack
        // bindings.
        let required_limits = adapter.limits();
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("origami-gpu"),
                required_features: wgpu::Features::empty(),
                required_limits,
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .map_err(|e| GpuInitError::DeviceRequest(e.to_string()))?;
        Ok(Self {
            instance,
            adapter,
            device,
            queue,
        })
    }

    /// Process-wide singleton.  Returns the same context on every
    /// call.  Initialisation errors are stored — a second call after
    /// a failed first call still returns the error.
    pub fn get() -> Result<&'static GpuContext, GpuInitError> {
        static CTX: OnceLock<Result<GpuContext, GpuInitError>> = OnceLock::new();
        CTX.get_or_init(Self::new).as_ref().map_err(|e| e.clone())
    }
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum GpuInitError {
    #[error("no GPU adapter found: {0}")]
    NoAdapter(String),
    #[error("device creation failed: {0}")]
    DeviceRequest(String),
}
