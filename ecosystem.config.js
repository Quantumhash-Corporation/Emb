module.exports = {
  apps: [{
    name: "bge-api",
    script: "/home/quantumhash/miniconda3/envs/emb/bin/uvicorn",
    interpreter: "none",
    cwd: "/home/quantumhash/LLM/bge-api",
    args: "app.main:app --host 0.0.0.0 --port 8100 --workers 1",
    env: {
      BGE_API_KEYS: "qun_2026_asdjsagdfdahdh",
      HF_TOKEN: "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",
      TRANSFORMERS_OFFLINE: "1",
      HF_DATASETS_OFFLINE: "1",
      PATH: "/home/quantumhash/miniconda3/envs/emb/bin:/usr/bin:/bin"
    },
    autorestart: true,
    watch: false,
    max_restarts: 5,
    restart_delay: 3000,
  }]
}
