# CPU Offload no Automanager

## Problema
O Automanager atualmente só suporta distribuição de carga entre GPUs. Quando um modelo GGUF excede a capacidade total de VRAM do(s) dispositivo(s), não há fallback para CPU. Além disso, o painel de "Recursos de GPU & Configuração" não mostra informações do processador nem da RAM do sistema.

## Solução Proposta
Adicionar suporte a CPU offload no Automanager, permitindo que o modelo use a CPU (via `--n-gpu-layers`) quando o VRAM total das GPUs for insuficiente.

## Escopo
- CPU aparece como mais uma linha na tabela "Recursos de GPU & Configuração"
- Nome do processador exibido (ex: "Intel Xeon E5-2680 v4")
- RAM do sistema exibida no padrão VRAM (usado / total em MB com barra)
- Monitoramento: CPU usage %
- Peso distribuído entre CPU e GPUs (soma total = 100%)
- Auto Balance inclui CPU
- Detecção automática do processador e RAM total

## Decisões Tomadas
- CPU aparece na mesma tabela das GPUs (mesmo padrão visual)
- Nome completo do processador exibido
- RAM no formato MB (igual VRAM)
- Peso compartilhado: CPU + GPUs = 100%
- Abordagem: CPU como Dispositivo Unificado
