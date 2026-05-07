use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use polars_core::prelude::*;

static REUSABLE_INPUT_LOCK: Mutex<()> = Mutex::new(());
static REUSABLE_INPUTS: OnceLock<Mutex<Option<Arc<HashMap<String, DataFrame>>>>> = OnceLock::new();

fn reusable_inputs() -> &'static Mutex<Option<Arc<HashMap<String, DataFrame>>>> {
    REUSABLE_INPUTS.get_or_init(|| Mutex::new(None))
}

struct ReusableInputsGuard {
    previous: Option<Arc<HashMap<String, DataFrame>>>,
}

impl Drop for ReusableInputsGuard {
    fn drop(&mut self) {
        let mut active = reusable_inputs().lock().unwrap();
        *active = self.previous.take();
    }
}

pub fn with_reusable_frame_inputs<F, T>(
    inputs: Arc<HashMap<String, DataFrame>>,
    f: F,
) -> PolarsResult<T>
where
    F: FnOnce() -> PolarsResult<T>,
{
    let _binding_guard = REUSABLE_INPUT_LOCK.lock().unwrap();
    let _guard = {
        let mut active = reusable_inputs().lock().unwrap();
        ReusableInputsGuard {
            previous: active.replace(inputs),
        }
    };

    f()
}

pub fn get_reusable_frame_input(source_id: &str) -> PolarsResult<DataFrame> {
    let active = reusable_inputs().lock().unwrap();
    let inputs = active.as_ref().ok_or_else(|| {
        polars_err!(
            ComputeError:
            "reusable source '{}' was collected without an active auto plan-cache input binding",
            source_id
        )
    })?;
    inputs
        .get(source_id)
        .cloned()
        .ok_or_else(|| polars_err!(ComputeError: "missing reusable source input: '{}'", source_id))
}

pub fn reusable_frame_inputs_active() -> bool {
    reusable_inputs().lock().unwrap().is_some()
}
