use super::*;

/// A batch factory invocation after CLI/config fields have been normalized.
/// Data adapters can evolve behind this boundary without changing model jobs.
#[derive(Debug)]
pub(super) struct BatchInput {
    pub entrypoint: Option<String>,
    pub parameters: Map<String, Value>,
    pub binding: Option<&'static str>,
}

pub(super) fn batch_input(inputs: &Inputs, operation: &str) -> Result<BatchInput, Report> {
    let parameters = parameters(&inputs.batch_params)
        .map_err(|error| Report::error(operation, "PARAMETER_INVALID", error, ""))?;
    if inputs
        .batch
        .as_deref()
        .is_some_and(|batch| batch.trim().is_empty())
    {
        return Err(Report::error(
            operation,
            "BATCH_ENTRYPOINT_INVALID",
            "批次入口不能为空。",
            "使用 package.module:factory，或移除空的 --batch 参数。",
        ));
    }
    if inputs.batch.is_none() && !parameters.is_empty() {
        return Err(Report::error(
            operation,
            "BATCH_PARAMS_WITHOUT_BATCH",
            "--batch-param 需要同时提供批次工厂。",
            "提供 --batch package.module:factory，或移除批次参数。",
        ));
    }
    if inputs.batch.is_none() && inputs.batch_binding.is_some() {
        return Err(Report::error(
            operation,
            "BATCH_BINDING_WITHOUT_BATCH",
            "批次绑定需要同时提供批次来源。",
            "提供 --batch package.module:factory，或移除 --batch-binding。",
        ));
    }
    let binding = inputs.batch_binding.as_ref().map(|binding| match binding {
        BatchBinding::Envelope => "envelope",
        BatchBinding::Single => "single",
        BatchBinding::SingleRaw => "single-raw",
        BatchBinding::Args => "args",
        BatchBinding::Kwargs => "kwargs",
    });
    Ok(BatchInput {
        entrypoint: inputs.batch.clone(),
        parameters,
        binding,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_invocation_normalizes_factory_parameters_and_binding() {
        let inputs = Inputs {
            batch: Some("project.data:sample".into()),
            batch_params: vec!["split=\"train\"".into()],
            batch_binding: Some(BatchBinding::Envelope),
            ..Inputs::default()
        };
        let batch = batch_input(&inputs, "probe").unwrap();
        assert_eq!(batch.entrypoint.as_deref(), Some("project.data:sample"));
        assert_eq!(batch.parameters["split"], json!("train"));
        assert_eq!(batch.binding, Some("envelope"));
    }

    #[test]
    fn invalid_batch_parameters_return_a_structured_application_diagnostic() {
        let inputs = Inputs {
            batch_params: vec!["split=train".into()],
            ..Inputs::default()
        };
        let error = batch_input(&inputs, "probe").unwrap_err();
        assert_eq!(error.operation, "probe");
        assert_eq!(error.diagnostics[0].code, "PARAMETER_INVALID");
    }

    #[test]
    fn batch_options_without_a_source_are_rejected_before_worker_dispatch() {
        let inputs = Inputs {
            batch_params: vec!["split=\"train\"".into()],
            ..Inputs::default()
        };
        let error = batch_input(&inputs, "probe").unwrap_err();
        assert_eq!(error.diagnostics[0].code, "BATCH_PARAMS_WITHOUT_BATCH");

        let inputs = Inputs {
            batch_binding: Some(BatchBinding::Args),
            ..Inputs::default()
        };
        let error = batch_input(&inputs, "probe").unwrap_err();
        assert_eq!(error.diagnostics[0].code, "BATCH_BINDING_WITHOUT_BATCH");
    }
}
