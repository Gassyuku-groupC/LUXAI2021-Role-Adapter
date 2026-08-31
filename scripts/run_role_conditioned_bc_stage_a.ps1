param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$Root = "D:\Luxai\Kaggle_Lux_AI_2021"
)

$ErrorActionPreference = "Stop"
Set-Location $Root
$index = ".\dataset\processed\best_c_weighted_bc\bc_index.csv"
$trainShards = ".\dataset\processed\best_c_role_bc_train"
$validationShards = ".\dataset\processed\best_c_role_bc_validation"
$output = ".\outputs\role_conditioned_bc_stageA"

New-Item -ItemType Directory -Force $trainShards, $validationShards, $output | Out-Null

& $Python .\scripts\extract_imitation_shards.py `
    --index $index `
    --agent-dir .\outputs\checkpoint_selection\agents\local_05120 `
    --output-dir $trainShards `
    --split train `
    --shard-size 4096
if ($LASTEXITCODE -ne 0) { throw "Train shard extraction failed." }

& $Python .\scripts\extract_imitation_shards.py `
    --index $index `
    --agent-dir .\outputs\checkpoint_selection\agents\local_05120 `
    --output-dir $validationShards `
    --split validation `
    --shard-size 4096
if ($LASTEXITCODE -ne 0) { throw "Validation shard extraction failed." }

& $Python .\scripts\train_role_conditioned_bc.py `
    --train-shards $trainShards `
    --validation-shards $validationShards `
    --agent-dir .\outputs\checkpoint_selection\agents\local_05120 `
    --output-dir $output `
    --epochs 3 `
    --batch-size 64 `
    --local-lr 5e-5 `
    --role-lr 5e-6 `
    --weight-decay 1e-4 `
    --delta-l2-cost 1e-3 `
    --log-interval 100 `
    --reset-local-projection
if ($LASTEXITCODE -ne 0) { throw "Role-conditioned BC failed." }
