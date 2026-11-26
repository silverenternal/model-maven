-- Pure capability/evidence normalization for the workspace preflight view.
-- Declarations describe the CLI or selected runtime. Only reports from the
-- current model context count as instance evidence.
local M = {}

local DECLARATIONS = {
  'probe.force_fresh_environment',
  'edit.conditional_source_plan',
  'edit.atomic_source_transaction',
  'layer.builtin_catalog',
  'training.conditional_config_start',
  'training.generic_loop_trial_cli_v1',
  'training.generic_loop_train_cli_v1',
  'training.conditional_supervision',
  'supervision.conditional_rule_add_v1',
  'supervision.observed_anchor_guard_v1',
}

local function is_list(value)
  return type(value) == 'table' and (vim.islist or vim.tbl_islist)(value)
end

local function lookup(capabilities, identifier)
  if type(capabilities) ~= 'table' then return nil end
  local value = capabilities[identifier]
  if value == nil and is_list(capabilities) then
    for _, descriptor in ipairs(capabilities) do
      if type(descriptor) == 'table' and descriptor.id == identifier then
        value = descriptor
        break
      end
    end
  end
  if value == true then return { state = 'declared' } end
  if value == false then
    -- Legacy interface maps only had positive booleans. A false value cannot
    -- distinguish an explicit rejection from an older/missing declaration.
    return { state = 'unknown', reason = 'Capability was not advertised.' }
  end
  if type(value) == 'table' then
    local state = value.state
    if state == 'supported' then return { state = 'declared', provider = value.provider, scope = value.scope } end
    if state == 'unsupported' then
      return { state = 'unsupported', reason = value.reason, provider = value.provider, scope = value.scope }
    end
    if state == 'conditional' then
      return { state = 'conditional', reason = value.reason, provider = value.provider, scope = value.scope }
    end
    return { state = 'unknown', reason = value.reason, provider = value.provider, scope = value.scope }
  end
  return nil
end

function M.summarize(capabilities, report, is_current, source_buffer)
  local declarations, missing = {}, {}
  for _, identifier in ipairs(DECLARATIONS) do
    local evidence = lookup(capabilities, identifier)
    if not evidence then evidence = { state = 'unknown', reason = 'No compatible capability declaration was returned.' } end
    declarations[#declarations + 1] = { id = identifier, scope = evidence.scope or 'cli_or_runtime', state = evidence.state,
      reason = evidence.reason, provider = evidence.provider }
    if evidence.state == 'unsupported' or evidence.state == 'conditional' or evidence.state == 'unknown' then
      missing[#missing + 1] = { id = identifier, state = evidence.state, reason = evidence.reason }
    end
  end

  local instance = { scope = 'model_instance', state = 'unknown', reason = 'No current instance report.' }
  if is_current == true and type(report) == 'table' and report.status == 'success' and type(report.data) == 'table' then
    if (report.operation == 'inspect' or report.operation == 'layers') and is_list(report.data.modules) then
      local total, source_set, source_insert = 0, 0, 0
      for _, module in ipairs(report.data.modules) do
        if type(module) == 'table' then
          total = total + 1
          if module.can_source_set == true then source_set = source_set + 1 end
          if module.can_source_insert == true then source_insert = source_insert + 1 end
        end
      end
      instance = { scope = 'model_instance', state = 'observed', kind = 'structure', module_count = total,
        source_set_candidates = source_set, source_insert_candidates = source_insert,
        reason = 'Candidates are report-specific; each requested edit still requires preview.' }
    elseif report.operation == 'probe' then
      local execution = report.data.forward_execution
      if type(report.data.evidence_cache) == 'table' and report.data.evidence_cache.status == 'reused' then
        execution = 'reused'
      end
      local state = execution == 'executed' and 'observed' or (execution == 'reused' and 'reused' or 'unknown')
      instance = { scope = 'model_instance', state = state, kind = 'forward_execution', execution = execution,
        reason = 'Evidence covers only observed inputs and paths; other branches remain unknown.' }
    elseif report.operation == 'training_probe' then
      local mode = report.data.training_mode
      local allowed_modes = { ['managed-step'] = true, ['manual-step'] = true, ['external-loop'] = true }
      local supervision = report.data.supervision
      local supervision_state = type(supervision) == 'table' and 'observed' or 'unknown'
      local supervision_status = type(supervision) == 'table' and supervision.status or nil
      local rule_count = type(supervision) == 'table' and is_list(supervision.rules) and #supervision.rules or nil
      local reported_capabilities = report.data.capabilities
      local training_seams = {}
      for _, name in ipairs({ 'forward_contract', 'loss_contract', 'backward_and_optimizer_updates', 'resume_cursor' }) do
        local value = type(reported_capabilities) == 'table' and reported_capabilities[name] or nil
        training_seams[name] = type(value) == 'string'
          and { state = 'reported', value = value }
          or { state = 'unknown' }
      end
      instance = {
        scope = 'model_instance', state = 'observed', kind = 'training_probe',
        training_mode = allowed_modes[mode] and mode or nil,
        training_mode_state = allowed_modes[mode] and 'observed' or 'unknown',
        training_seams = training_seams,
        reported_coverage_limit = type(report.data.coverage) == 'string' and report.data.coverage or nil,
        supervision = {
          state = supervision_state,
          status = supervision_status,
          rule_count = rule_count,
          coverage_status = type(supervision) == 'table' and supervision.coverage_status or nil,
        },
        reason = 'This is evidence from the reported trial only; it does not establish behavior for other inputs or unreported supervision seams.',
      }
    end
  elseif is_current ~= true then
    instance.reason = 'The report is stale or belongs to another model context.'
  end

  return {
    declarations = declarations,
    missing = missing,
    instance = instance,
    manual_source = {
      available = type(source_buffer) == 'number' and source_buffer >= 0,
      source_buffer = source_buffer,
      requires_preview = true,
      requires_revalidation = true,
    },
  }
end

-- Fail closed for automatic source edits. `preview.safe` is deliberately a
-- normalized caller assertion: the workspace must set it only after a fresh,
-- successful preview for this exact candidate and current source revision.
-- Keep the caller's buffer handle in the result so manual takeover can return
-- to the same source buffer when a capability or safety condition is missing.
function M.edit_decision(capabilities, candidate, preview, is_current, source_buffer)
  local plan = lookup(capabilities, 'edit.conditional_source_plan')
  local transaction = lookup(capabilities, 'edit.atomic_source_transaction')
  plan = plan or { state = 'unknown', reason = 'No source-plan declaration was returned.' }
  transaction = transaction or { state = 'unknown', reason = 'No source-transaction declaration was returned.' }
  local capability_feedback = {
    { id = 'edit.conditional_source_plan', state = plan.state, reason = plan.reason },
    { id = 'edit.atomic_source_transaction', state = transaction.state, reason = transaction.reason },
  }
  local declared = plan and plan.state == 'declared' and transaction and transaction.state == 'declared'
  local candidate_ok = type(candidate) == 'table' and candidate.state == 'observed'
    and candidate.can_edit == true
  local preview_ok = type(preview) == 'table' and preview.safe == true and preview.current == true
  local current = is_current == true
  local has_source_buffer = type(source_buffer) == 'number' and source_buffer >= 0
  if declared and candidate_ok and preview_ok and current then
    return { mode = 'automatic_candidate', state = 'conditional', source_buffer = source_buffer,
      capability_feedback = capability_feedback,
      reason = 'The exact edit still requires the normal apply confirmation and revision guard.' }
  end

  local reason
  if not current then reason = 'Model or source evidence is stale; automatic editing is not authorized.'
  elseif not declared then reason = 'The CLI did not declare the required guarded source-edit capabilities.'
  elseif not candidate_ok then reason = 'This model instance did not report an editable source candidate.'
  else reason = 'A fresh successful preview for this candidate and source revision is missing.' end
  return { mode = 'manual_source', state = has_source_buffer and 'available' or 'unknown',
    handoff_state = has_source_buffer and 'ready' or 'needs_source_buffer', source_buffer = source_buffer, reason = reason,
    capability_feedback = capability_feedback, requires_revalidation = true }
end

return M
