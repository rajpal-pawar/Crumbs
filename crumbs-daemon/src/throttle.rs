//! Platform-specific process throttling.
//!
//! The goal is to make `crumbs-daemon` essentially invisible to the user on
//! the target hardware (i7-6500U, 8 GB RAM).  We lower both CPU scheduling
//! priority and I/O priority so that latency-sensitive foreground work is
//! never starved.
//!
//! # Platform implementations
//!
//! | Platform | CPU priority | I/O priority |
//! |---|---|---|
//! | Windows  | `SetPriorityClass(PROCESS_MODE_BACKGROUND_BEGIN)` | handled automatically by Windows when background mode is set |
//! | Linux    | `setpriority(PRIO_PROCESS, 0, 19)` | `ioprio_set(IOPRIO_WHO_PROCESS, 0, IOPRIO_CLASS_IDLE)` via `syscall` |
//!
//! On unsupported platforms (macOS, etc.) `apply()` is a no-op that succeeds.

/// Lowers CPU and I/O priority of the current process to background levels.
///
/// # Errors
/// Returns an error string describing what failed.  Callers should treat a
/// throttle failure as **non-fatal** — warn and continue.
pub fn apply() -> Result<(), String> {
    _apply_inner()
}

// ---------------------------------------------------------------------------
// Windows implementation
// ---------------------------------------------------------------------------
#[cfg(windows)]
fn _apply_inner() -> Result<(), String> {
    use windows::Win32::Foundation::HANDLE;
    use windows::Win32::System::Threading::{
        GetCurrentProcess, SetPriorityClass, PROCESS_MODE_BACKGROUND_BEGIN,
    };

    // SAFETY: `GetCurrentProcess` returns a pseudo-handle that is always
    // valid for the lifetime of the process and does not need to be closed.
    let handle: HANDLE = unsafe { GetCurrentProcess() };

    // PROCESS_MODE_BACKGROUND_BEGIN lowers both CPU and disk I/O priority in
    // one call.  It is the recommended way to be a "good citizen" on Windows.
    // See: https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-setpriorityclass
    unsafe {
        SetPriorityClass(handle, PROCESS_MODE_BACKGROUND_BEGIN).map_err(|e| {
            format!(
                "SetPriorityClass(PROCESS_MODE_BACKGROUND_BEGIN) failed: {e}"
            )
        })?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Linux implementation
// ---------------------------------------------------------------------------
#[cfg(target_os = "linux")]
fn _apply_inner() -> Result<(), String> {
    apply_cpu_priority()?;
    apply_io_priority()?;
    Ok(())
}

#[cfg(target_os = "linux")]
fn apply_cpu_priority() -> Result<(), String> {
    // setpriority(PRIO_PROCESS, 0 /*self*/, 19 /*lowest nice value*/)
    // nice(19) is the highest "niceness" (lowest priority) on Linux.
    let ret = unsafe {
        libc::setpriority(
            libc::PRIO_PROCESS,
            0, // pid = 0 means the calling process
            19,
        )
    };
    if ret == -1 {
        let errno = unsafe { *libc::__errno_location() };
        return Err(format!(
            "setpriority(PRIO_PROCESS, 0, 19) failed: errno={errno}"
        ));
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn apply_io_priority() -> Result<(), String> {
    // ioprio_set is not wrapped by libc, so we call it through libc::syscall.
    //
    // Kernel constants (from <linux/ioprio.h>):
    //   IOPRIO_CLASS_IDLE = 3
    //   IOPRIO_PRIO_VALUE(class, data) = ((class) << 13) | (data)
    //   IOPRIO_WHO_PROCESS = 1
    //   SYS_ioprio_set = 251 (x86_64)
    //
    // We want: IOPRIO_PRIO_VALUE(IOPRIO_CLASS_IDLE, 0) = (3 << 13) | 0 = 24576

    const IOPRIO_WHO_PROCESS: libc::c_int = 1;
    const IOPRIO_CLASS_IDLE: libc::c_int = 3;
    const IOPRIO_CLASS_SHIFT: libc::c_int = 13;
    const IOPRIO_PRIO_VALUE: libc::c_int = (IOPRIO_CLASS_IDLE << IOPRIO_CLASS_SHIFT) | 0;

    let ret = unsafe {
        libc::syscall(
            libc::SYS_ioprio_set,
            IOPRIO_WHO_PROCESS as libc::c_long,
            0 as libc::c_long, // pid = 0 means the calling process/thread
            IOPRIO_PRIO_VALUE as libc::c_long,
        )
    };
    if ret == -1 {
        let errno = unsafe { *libc::__errno_location() };
        return Err(format!(
            "ioprio_set(IOPRIO_WHO_PROCESS, 0, IOPRIO_CLASS_IDLE) failed: errno={errno}"
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Fallback for all other platforms (macOS, etc.)
// ---------------------------------------------------------------------------
#[cfg(not(any(windows, target_os = "linux")))]
fn _apply_inner() -> Result<(), String> {
    // No-op on unsupported platforms.
    Ok(())
}
