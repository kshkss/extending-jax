use std::collections::HashMap;
use std::ffi::c_void;

use pyo3::prelude::*;

use num_traits::Float;

#[inline]
fn sincos<T: Float>(x: &T, sx: &mut T, cx: &mut T) {
    *sx = x.sin();
    *cx = x.cos();
}

#[inline]
fn compute_eccentric_anomaly<T: Float>(
    mean_anom: &T,
    ecc: &T,
    sin_ecc_anom: &mut T,
    cos_ecc_anom: &mut T,
) {
    let tol = T::from(1e-12).unwrap();
    let mut e = if mean_anom < &T::from(std::f64::consts::PI).unwrap() {
        *mean_anom + *ecc * T::from(0.85).unwrap()
    } else {
        *mean_anom - *ecc * T::from(0.85).unwrap()
    };
    for _i in 0..20 {
        sincos(&e, sin_ecc_anom, cos_ecc_anom);
        let g = e - *ecc * *sin_ecc_anom - *mean_anom;
        if g.abs() <= tol {
            return;
        }
        e = e - g / (T::one() - *ecc * *cos_ecc_anom);
    }
}

pub extern "C" fn cpu_kepler<T: Float>(out_tuple: *mut c_void, in_: *const *const c_void) {
    // Parse the inputs
    let in_ = unsafe { std::slice::from_raw_parts(in_, 3) };
    let size = unsafe { *(in_[0] as *const i64) } as usize;
    let mean_anom = unsafe { std::slice::from_raw_parts(in_[1] as *const T, size) };
    let ecc = unsafe { std::slice::from_raw_parts(in_[2] as *const T, size) };

    // The output is stored as a list of pointers since we have multiple outputs
    let out = unsafe { std::slice::from_raw_parts(out_tuple as *const *mut T, 2) };
    let sin_ecc_anom = unsafe { std::slice::from_raw_parts_mut(out[0], size) };
    let cos_ecc_anom = unsafe { std::slice::from_raw_parts_mut(out[1], size) };

    for n in 0..size {
        compute_eccentric_anomaly(
            &mean_anom[n],
            &ecc[n],
            &mut sin_ecc_anom[n],
            &mut cos_ecc_anom[n],
        );
    }
}

/// Prints a message.
#[pyfunction]
fn registrations() -> PyResult<HashMap<String, fn(*mut c_void, *const *const c_void)>> {
    let mut dict = HashMap::new();
    Ok(dict)
}

/// A Python module implemented in Rust.
#[pymodule]
fn cpu_ops(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(registrations, m)?)?;
    Ok(())
}
