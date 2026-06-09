"""Testes da cascata estrita por prioridade do LoadDistributor (task_01).

Valida os cenários A–D do PRD (preencher cada GPU até 98% antes da próxima;
CPU por último) e as bordas (CPU desligada infeasível, GPU única, arredondamento,
zero-offload com VRAM ociosa).

Convenção dos testes: VRAM e tamanho do modelo em "unidades" inteiras (mesma
escala). 3090 = 24000, P100 = 16000. Limite de 98% → caps 23520 / 15680.
"""

import pytest

from load_distributor import DistributionResult, LoadDistributor

# Hardware de referência do PRD.
GPU3090 = 24000
P100 = 16000
CAP_3090 = int(GPU3090 * 0.98)  # 23520
CAP_P100 = int(P100 * 0.98)     # 15680


def _three_gpus():
    """3090 (principal, idx 0) + 2× P100 (idx 1, 2)."""
    return {0: GPU3090, 1: P100, 2: P100}


def _distribute(model_mb, cpu_enabled=True, vram=None, order=None):
    vram = vram if vram is not None else _three_gpus()
    order = order if order is not None else sorted(vram)
    return LoadDistributor.distribute(
        gpu_vram=vram,
        priority_order=order,
        estimated_model_vram_mb=model_mb,
        cpu_enabled=cpu_enabled,
    )


def _assert_sums_to_100(r: DistributionResult):
    assert sum(r.gpu_weights.values()) + r.cpu_weight == 100


# ---------------------------------------------------------------------------
# Cenários A–D do PRD
# ---------------------------------------------------------------------------

def test_cenario_a_modelo_cabe_na_principal():
    """A: modelo 20GB → 3090=100%, P100s=0, CPU=0."""
    r = _distribute(20000)
    assert r.is_feasible
    assert r.gpu_weights == {0: 100, 1: 0, 2: 0}
    assert r.cpu_weight == 0
    _assert_sums_to_100(r)


def test_cenario_b_transborda_para_segunda_gpu():
    """B: modelo 30GB → 3090 cheia, P100#1 com a sobra, P100#2=0, CPU=0."""
    r = _distribute(30000)
    assert r.is_feasible
    assert r.cpu_weight == 0
    assert r.gpu_weights[2] == 0            # terceira GPU intocada
    assert r.gpu_weights[1] > 0             # segunda recebe a sobra
    assert r.gpu_weights[0] > r.gpu_weights[1]
    _assert_sums_to_100(r)


def test_cenario_c_usa_tres_gpus_sem_cpu():
    """C: modelo 50GB → 3090 + P100#1 cheias, P100#2 com sobra, CPU=0."""
    r = _distribute(50000)
    assert r.is_feasible
    assert r.cpu_weight == 0
    assert all(r.gpu_weights[i] > 0 for i in (0, 1, 2))
    _assert_sums_to_100(r)


def test_cenario_d_transborda_para_cpu():
    """D: modelo 70GB → todas as GPUs a 98%, CPU com a sobra."""
    r = _distribute(70000)
    assert r.is_feasible
    assert r.cpu_weight > 0
    assert all(r.gpu_weights[i] > 0 for i in (0, 1, 2))
    # CPU recebe ~15120/70000 ≈ 21-22%.
    assert 20 <= r.cpu_weight <= 23
    _assert_sums_to_100(r)


# ---------------------------------------------------------------------------
# Bordas
# ---------------------------------------------------------------------------

def test_cpu_desligada_e_modelo_nao_cabe_infeasivel():
    """CPU off + modelo > soma das GPUs → is_feasible=False."""
    r = _distribute(70000, cpu_enabled=False)
    assert r.is_feasible is False
    assert r.cpu_weight == 0


def test_cpu_desligada_e_modelo_cabe_feasivel():
    """CPU off + modelo cabe nas GPUs → feasível, sem CPU."""
    r = _distribute(50000, cpu_enabled=False)
    assert r.is_feasible
    assert r.cpu_weight == 0


def test_gpu_unica_recebe_100():
    r = LoadDistributor.distribute(
        gpu_vram={0: GPU3090},
        priority_order=[0],
        estimated_model_vram_mb=10000,
        cpu_enabled=True,
    )
    assert r.gpu_weights == {0: 100}
    assert r.cpu_weight == 0
    assert r.is_feasible


def test_sem_gpus_infeasivel():
    r = LoadDistributor.distribute(
        gpu_vram={}, priority_order=[], estimated_model_vram_mb=10000
    )
    assert r.is_feasible is False
    assert r.gpu_weights == {}


def test_modelo_de_tamanho_desconhecido_mantem_gpus():
    """estimated<=0 → passthrough dos pesos sem CPU."""
    r = LoadDistributor.distribute(
        gpu_vram={0: GPU3090, 1: P100},
        gpu_weights={0: 60, 1: 40},
        estimated_model_vram_mb=0,
        cpu_enabled=True,
    )
    assert r.is_feasible
    assert r.cpu_weight == 0


# ---------------------------------------------------------------------------
# Ordem de prioridade
# ---------------------------------------------------------------------------

def test_prioridade_preenche_principal_primeiro():
    """Principal = idx 2 (P100); modelo cabe nela → GPU 2=100%, demais 0."""
    r = _distribute(10000, order=[2, 0, 1])  # 10000 < cap P100 (15680)
    assert r.gpu_weights[2] == 100
    assert r.gpu_weights[0] == 0
    assert r.gpu_weights[1] == 0


def test_prioridade_transborda_na_ordem_dada():
    """Principal = idx 2; modelo 30GB → 2 cheia, 0 com sobra, 1 intocada."""
    r = _distribute(30000, order=[2, 0, 1])
    assert r.gpu_weights[2] > r.gpu_weights[0] > 0
    assert r.gpu_weights[1] == 0
    assert r.cpu_weight == 0


# ---------------------------------------------------------------------------
# Invariantes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_mb", [20000, 30000, 50000, 70000, 100000])
def test_soma_sempre_100(model_mb):
    r = _distribute(model_mb)
    _assert_sums_to_100(r)


@pytest.mark.parametrize("model_mb", [1000, 20000, 39200])  # <= soma dos caps
def test_zero_offload_enquanto_ha_vram_ociosa(model_mb):
    """Nenhum peso de CPU enquanto o modelo couber na soma das GPUs (caps)."""
    r = _distribute(model_mb)
    assert r.cpu_weight == 0
    assert r.is_feasible


def test_arredondamento_vram_desigual_fecha_em_100():
    """VRAM bem desigual + transbordo p/ CPU → soma exata de 100%."""
    r = LoadDistributor.distribute(
        gpu_vram={0: 11000, 1: 7000},
        priority_order=[0, 1],
        estimated_model_vram_mb=30000,
        cpu_enabled=True,
    )
    _assert_sums_to_100(r)
    assert r.cpu_weight > 0
    assert r.gpu_weights[0] > r.gpu_weights[1] > 0
