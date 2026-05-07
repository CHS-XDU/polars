use std::collections::{HashMap, VecDeque};
use std::fmt::Write;
use std::sync::{Arc, Mutex, OnceLock};

use polars_core::prelude::*;
use polars_core::query_result::QueryResult;
use polars_expr::state::ExecutionState;
use polars_mem_engine::{Executor, create_physical_plan};
use polars_plan::prelude::*;
use polars_utils::pl_str::PlSmallStr;

use super::BUILD_STREAMING_EXECUTOR;

const SOURCE_PREFIX: &str = "__polars_auto_df_";

type PhysicalPlan = Arc<Mutex<Box<dyn Executor>>>;

struct PlanCache {
    max_entries: usize,
    order: VecDeque<String>,
    plans: HashMap<String, PhysicalPlan>,
}

impl PlanCache {
    fn new() -> Self {
        let max_entries = std::env::var("POLARS_AUTO_PLAN_CACHE_MAX_ENTRIES")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(32);
        Self {
            max_entries,
            order: VecDeque::new(),
            plans: HashMap::new(),
        }
    }

    fn get(&mut self, key: &str) -> Option<PhysicalPlan> {
        let plan = self.plans.get(key).cloned()?;
        self.touch(key);
        Some(plan)
    }

    fn insert(&mut self, key: String, plan: PhysicalPlan) -> PhysicalPlan {
        self.plans.insert(key.clone(), Arc::clone(&plan));
        self.touch(&key);
        while self.plans.len() > self.max_entries {
            if let Some(old_key) = self.order.pop_front() {
                self.plans.remove(&old_key);
            } else {
                break;
            }
        }
        plan
    }

    fn touch(&mut self, key: &str) {
        if let Some(position) = self.order.iter().position(|existing| existing == key) {
            self.order.remove(position);
        }
        self.order.push_back(key.to_string());
    }
}

static PLAN_CACHE: OnceLock<Mutex<PlanCache>> = OnceLock::new();

pub(super) struct DslCacheSeed {
    key: String,
    inputs: Arc<HashMap<String, DataFrame>>,
    source_ids_by_df_ptr: HashMap<usize, PlSmallStr>,
}

struct RewrittenScans {
    inputs: HashMap<String, DataFrame>,
    source_signature: String,
}

fn plan_cache() -> &'static Mutex<PlanCache> {
    PLAN_CACHE.get_or_init(|| Mutex::new(PlanCache::new()))
}

fn enabled() -> bool {
    std::env::var("POLARS_AUTO_PLAN_CACHE").as_deref() == Ok("1")
}

fn trace_enabled() -> bool {
    std::env::var("POLARS_AUTO_PLAN_CACHE_TRACE").as_deref() == Ok("1")
}

fn row_bucket(n_rows: usize) -> (usize, usize) {
    if n_rows == 0 {
        return (0, 0);
    }

    let mut upper = 1024usize;
    while upper < n_rows {
        upper = upper.saturating_mul(2);
        if upper == usize::MAX {
            break;
        }
    }
    let lower = if upper == 1024 { 1 } else { (upper / 2) + 1 };
    (lower, upper)
}

fn dataframe_ptr(df: &Arc<DataFrame>) -> usize {
    Arc::as_ptr(df) as usize
}

fn write_schema_signature(buffer: &mut String, schema: &Schema) {
    for (name, dtype) in schema.iter() {
        let _ = write!(buffer, "{}:{:?};", name, dtype);
    }
}

fn source_id_for_scan(scan_idx: usize) -> PlSmallStr {
    PlSmallStr::from_string(format!("{SOURCE_PREFIX}{scan_idx}"))
}

fn append_source_signature(
    buffer: &mut String,
    source_id: &str,
    n_rows: usize,
    schema: &Schema,
) -> (usize, usize) {
    let (min_rows, max_rows) = row_bucket(n_rows);
    let _ = write!(buffer, "{}:{}-{}:", source_id, min_rows, max_rows);
    write_schema_signature(buffer, schema);
    buffer.push('|');
    (min_rows, max_rows)
}

fn trace(message: &str) {
    if trace_enabled() {
        eprintln!("POLARS_AUTO_PLAN_CACHE {message}");
    }
}

pub(super) fn try_build_seed(
    plan: &DslPlan,
    opt_state: OptFlags,
) -> PolarsResult<Option<DslCacheSeed>> {
    if !enabled() || polars_plan::plans::reusable_scan::reusable_frame_inputs_active() {
        return Ok(None);
    }

    let mut inputs = HashMap::new();
    let mut source_ids_by_df_ptr = HashMap::new();
    let mut source_signature = String::new();
    let mut scan_idx = 0usize;

    for dsl in plan {
        match dsl {
            DslPlan::DataFrameScan { df, schema } => {
                let source_id = source_id_for_scan(scan_idx);
                let source_id_string = source_id.to_string();
                append_source_signature(
                    &mut source_signature,
                    &source_id_string,
                    df.height(),
                    schema.as_ref(),
                );

                inputs.insert(source_id_string, df.as_ref().clone());
                source_ids_by_df_ptr.insert(dataframe_ptr(df), source_id);
                scan_idx += 1;
            },
            DslPlan::Scan { .. } => return Ok(None),
            #[cfg(feature = "python")]
            DslPlan::PythonScan { .. } => return Ok(None),
            _ => {},
        }
    }

    if inputs.is_empty() {
        return Ok(None);
    }

    let key = format!(
        "polars-auto-plan-cache-dsl-v1\ncrate={}\nopt_flags={}\nsources={}\n{}",
        env!("CARGO_PKG_VERSION"),
        opt_state.bits(),
        source_signature,
        plan.describe()?
    );

    Ok(Some(DslCacheSeed {
        key,
        inputs: Arc::new(inputs),
        source_ids_by_df_ptr,
    }))
}

fn rewrite_dataframe_scans(
    ir_plan: &mut IRPlan,
    source_ids_by_df_ptr: Option<&HashMap<usize, PlSmallStr>>,
) -> PolarsResult<Option<RewrittenScans>> {
    let nodes = ir_plan
        .lp_arena
        .iter(ir_plan.lp_top)
        .map(|(node, _)| node)
        .collect::<Vec<_>>();
    let mut inputs = HashMap::new();
    let mut source_signature = String::new();
    let mut scan_idx = 0usize;

    for node in nodes {
        let replacement = match ir_plan.lp_arena.get(node) {
            IR::DataFrameScan {
                df,
                schema,
                output_schema,
            } => {
                let source_id = if let Some(source_ids_by_df_ptr) = source_ids_by_df_ptr {
                    let df_ptr = dataframe_ptr(df);
                    let Some(source_id) = source_ids_by_df_ptr.get(&df_ptr) else {
                        return Ok(None);
                    };
                    source_id.clone()
                } else {
                    source_id_for_scan(scan_idx)
                };
                let source_id_string = source_id.to_string();
                let (min_rows, max_rows) = append_source_signature(
                    &mut source_signature,
                    &source_id_string,
                    df.height(),
                    schema.as_ref(),
                );

                inputs.insert(source_id_string, df.as_ref().clone());
                scan_idx += 1;

                Some(IR::ReusableDataFrameScan {
                    source_id,
                    schema: schema.clone(),
                    output_schema: output_schema.clone(),
                    min_rows: Some(min_rows),
                    max_rows: Some(max_rows),
                })
            },
            IR::Scan { .. } => return Ok(None),
            #[cfg(feature = "python")]
            IR::PythonScan { .. } => return Ok(None),
            IR::ReusableDataFrameScan { .. } => return Ok(None),
            _ => None,
        };

        if let Some(replacement) = replacement {
            ir_plan.lp_arena.replace(node, replacement);
        }
    }

    if inputs.is_empty() {
        Ok(None)
    } else {
        Ok(Some(RewrittenScans {
            inputs,
            source_signature,
        }))
    }
}

fn fallback_key(ir_plan: &IRPlan, source_signature: &str) -> String {
    format!(
        "polars-auto-plan-cache-v1\ncrate={}\nsources={}\n{}",
        env!("CARGO_PKG_VERSION"),
        source_signature,
        ir_plan.describe()
    )
}

fn get_or_create_physical_plan(key: String, template_plan: &IRPlan) -> PolarsResult<PhysicalPlan> {
    {
        let mut cache = plan_cache().lock().unwrap();
        if let Some(plan) = cache.get(&key) {
            trace("hit");
            return Ok(plan);
        }
    }

    trace("miss");

    let mut physical_ir_plan = template_plan.clone();
    let physical_plan = create_physical_plan(
        physical_ir_plan.lp_top,
        &mut physical_ir_plan.lp_arena,
        &mut physical_ir_plan.expr_arena,
        BUILD_STREAMING_EXECUTOR,
    )?;
    let physical_plan = Arc::new(Mutex::new(physical_plan));

    let mut cache = plan_cache().lock().unwrap();
    Ok(cache.insert(key, physical_plan))
}

fn execute_with_inputs(
    physical_plan: PhysicalPlan,
    inputs: Arc<HashMap<String, DataFrame>>,
) -> PolarsResult<QueryResult> {
    polars_plan::plans::reusable_scan::with_reusable_frame_inputs(inputs, || {
        let mut state = ExecutionState::new();
        physical_plan
            .lock()
            .unwrap()
            .execute(&mut state)
            .map(QueryResult::Single)
    })
}

pub(super) fn try_execute_seed(seed: Option<&DslCacheSeed>) -> PolarsResult<Option<QueryResult>> {
    let Some(seed) = seed else {
        return Ok(None);
    };

    let physical_plan = {
        let mut cache = plan_cache().lock().unwrap();
        let Some(plan) = cache.get(&seed.key) else {
            return Ok(None);
        };
        trace("hit");
        plan
    };

    execute_with_inputs(physical_plan, Arc::clone(&seed.inputs)).map(Some)
}

pub(super) fn try_collect(
    ir_plan: &IRPlan,
    seed: Option<&DslCacheSeed>,
) -> PolarsResult<Option<QueryResult>> {
    if !enabled()
        || polars_plan::plans::reusable_scan::reusable_frame_inputs_active()
        || matches!(ir_plan.root(), IR::SinkMultiple { .. })
    {
        return Ok(None);
    }

    let mut template_plan = ir_plan.clone();
    let Some(rewritten) = rewrite_dataframe_scans(
        &mut template_plan,
        seed.map(|seed| &seed.source_ids_by_df_ptr),
    )?
    else {
        return Ok(None);
    };

    let (key, inputs) = seed
        .map(|seed| (seed.key.clone(), Arc::clone(&seed.inputs)))
        .unwrap_or_else(|| {
            (
                fallback_key(&template_plan, &rewritten.source_signature),
                Arc::new(rewritten.inputs),
            )
        });
    let physical_plan = get_or_create_physical_plan(key, &template_plan)?;

    execute_with_inputs(physical_plan, inputs).map(Some)
}
