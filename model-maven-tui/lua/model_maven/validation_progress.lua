local M = {}

local locale = require('model_maven.locale')
local tr = locale.text

local function line(english, simplified, traditional)
  return tr(english, simplified, traditional)
end

local function clean(value, limit)
  if type(value) ~= 'string' then return nil end
  return vim.fn.strcharpart(value:gsub('[\r\n%z\1-\31\127]', ' '), 0, limit)
end

-- Describe only evidence passed by the caller. In particular, targeted
-- replay still requires a full root-model forward in the current backend.
function M.cost_lines(mode, evidence)
  evidence = type(evidence) == 'table' and evidence or {}
  local lines = {}

  if mode == 'static' then
    lines[#lines + 1] = line(
      'Static inspection reads reported structure and configuration without executing the model; if the operation imports user code, it requires prior confirmation.',
      '静态查看读取已报告的结构和配置，不执行模型；如果操作会导入用户代码，须事先确认。',
      '靜態檢視讀取已回報的結構和設定，不執行模型；若操作會匯入使用者程式碼，須事先確認。')
    lines[#lines + 1] = line(
      'User-code imports, runtime dependencies, and resource cost: unknown unless separately reported.',
      '用户代码导入、运行时依赖和资源成本：除非另有报告，否则未知。',
      '使用者程式碼匯入、執行期相依項和資源成本：除非另有回報，否則未知。')
  elseif mode == 'targeted' then
    if evidence.supported ~= true then
      lines[#lines + 1] = line(
        'Local replay is unavailable: a supported boundary and valid observed input were not confirmed.',
        '局部重放不可用：尚未确认受支持的边界和有效的已观察输入。',
        '局部重播不可用：尚未確認受支援的邊界和有效的已觀察輸入。')
    else
      lines[#lines + 1] = line(
        'Scope: selected-module replay. The current path first constructs the model and executes a full root-model forward to capture the call.',
        '范围：选中模块重放。当前路径会先构造模型并执行完整根模型前向以捕获调用。',
        '範圍：選取模組重播。目前路徑會先建構模型並執行完整根模型前向以擷取呼叫。')
      lines[#lines + 1] = line(
        'The module copy includes its loaded state; checkpoint loading, backward propagation, updates, and external side effects are not isolated.',
        '模块副本包含其已加载状态；不隔离检查点加载、反向传播、参数更新或外部副作用。',
        '模組副本包含其已載入狀態；不隔離檢查點載入、反向傳播、參數更新或外部副作用。')
    end
    lines[#lines + 1] = line(
      'Elapsed time and peak resource use: not measured.',
      '耗时和峰值资源使用：未测量。',
      '耗時和峰值資源使用：未測量。')
  elseif mode == 'failed_forward' then
    lines[#lines + 1] = line(
      'The selected model and input reached a forward pass that failed. The recorded calls cover only execution before the failure.',
      '所选模型和输入已进入前向，但前向失败；调用记录只覆盖失败前的执行。',
      '所選模型和輸入已進入前向，但前向失敗；呼叫紀錄只涵蓋失敗前的執行。')
  elseif mode == 'full' then
    lines[#lines + 1] = line(
      'Full validation executes the selected model and input; confirm the reported environment and possible model, weight, and device costs before starting.',
      '完整验证会执行所选模型和输入；启动前请确认已报告的环境，以及可能涉及的模型、权重和设备成本。',
      '完整驗證會執行所選模型和輸入；開始前請確認已回報的環境，以及可能涉及的模型、權重和裝置成本。')
    lines[#lines + 1] = line(
      'Only the reported input, execution path, weights, and environment are covered; missing facts remain unknown.',
      '仅覆盖报告中的输入、执行路径、权重和环境；缺失事实保持未知。',
      '僅涵蓋回報中的輸入、執行路徑、權重和環境；缺少的事實維持未知。')
  else
    lines[#lines + 1] = line(
      'Validation mode is unknown; no execution scope or cost is inferred.',
      '验证模式未知；不推断执行范围或成本。',
      '驗證模式未知；不推斷執行範圍或成本。')
  end

  local dependencies = {}
  if type(evidence.heavy_dependencies) == 'table' then
    for index, dependency in ipairs(evidence.heavy_dependencies) do
      local value = clean(dependency, 120)
      if value and value ~= '' then dependencies[#dependencies + 1] = value end
      if index >= 16 then break end
    end
  end
  if #dependencies > 0 then
    lines[#lines + 1] = line('Reported dependencies: ', '已报告依赖：', '已回報相依項：')
      .. table.concat(dependencies, ', ')
  else
    lines[#lines + 1] = line(
      'Heavy dependencies and resource requirements: unknown.',
      '重量级依赖和资源需求：未知。',
      '重量級相依項和資源需求：未知。')
  end
  return lines
end

-- The caller must pass an observed transport/process state; this helper never
-- infers completion or cancellation from a missing report.
function M.progress_lines(progress)
  progress = type(progress) == 'table' and progress or {}
  local labels = {
    running = line('Running', '运行中', '執行中'),
    waiting = line('Waiting', '等待中', '等待中'),
    cancel_requested = line('Cancellation requested; waiting for process status', '已请求取消；正在等待进程状态', '已要求取消；正在等待程序狀態'),
    stop_requested = line('Stop requested; waiting for final status', '已请求停止；等待最终状态', '已要求停止；等待最終狀態'),
    completed = line('Completed', '已完成', '已完成'),
    failed = line('Failed', '失败', '失敗'),
    interrupted = line('Interrupted', '已中断', '已中斷'),
    cancelled = line('Cancelled', '已取消', '已取消'),
    unknown = line('Status unknown', '状态未知', '狀態未知'),
  }
  local state = type(progress.state) == 'string' and progress.state or 'unknown'
  local lines = { line('Status: ', '状态：', '狀態：') .. (labels[state] or labels.unknown) }

  if type(progress.elapsed_seconds) == 'number' and progress.elapsed_seconds >= 0 then
    lines[#lines + 1] = line('Elapsed: ', '已用时：', '已用時：')
      .. string.format('%.1f s', progress.elapsed_seconds)
  end
  if progress.cancellable == true then
    lines[#lines + 1] = line('Cancellation is available.', '可以请求取消。', '可以要求取消。')
  elseif progress.cancellable == false then
    lines[#lines + 1] = line('Cancellation is unavailable.', '当前不可取消。', '目前不可取消。')
  else
    lines[#lines + 1] = line('Cancellation availability: unknown.', '是否可取消：未知。', '是否可取消：未知。')
  end
  local detail = clean(progress.detail, 512)
  if detail and detail ~= '' then
    lines[#lines + 1] = line('Details: ', '详情：', '詳細資料：') .. detail
  end
  return lines
end

return M
