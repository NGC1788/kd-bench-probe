# kd-bench-probe

**정답 라벨(CE)이 teacher 마스킹의 손해를 가려 주는가?** — [KSHS × AIM Lab 벤치마크](https://github.com/jeehoo0507/kshs-aimlab-benchmarks)에서 확인하는 실험 코드.

*Does ground-truth CE hide the cost of teacher-side masking (MaskedKD)? A probe on the KSHS × AIM Lab benchmark, using its reference trainer unchanged.*

## 한 폴더에 전부
- 설치되는 것은 **전부 이 폴더 안에** 생겨요: uv가 관리하는 Python 3.11, `.venv`, 패키지 캐시, DeiT 가중치, CUDA 커널 캐시, 벤치마크 코드, 데이터셋, 결과.
- `~/.cache`, `~/.local`, 셸 설정 파일, 시스템 Python은 **건드리지 않아요**. 설정은 [`env.sh`](env.sh)에 모여 있어요.
- 서버에 `uv`가 없으면 공식 설치 스크립트로 `./.tools/uv`에만 설치해요(PATH와 셸 설정 파일은 바꾸지 않음).
- **다 지우려면 이 폴더를 삭제하면 끝**이에요.

```bash
rm -rf kd-bench-probe
```

## 실행 (GPU 서버)
```bash
git clone https://github.com/NGC1788/kd-bench-probe && cd kd-bench-probe
./run.sh setup                    # uv, Python 3.11, torch 2.5.1+cu124(기준 run과 같음), 벤치마크 @ e1c39e7
./run.sh all --dataset coco       # 데이터 준비 → teacher 30 epoch → 4조건 × 3 seed → 결과 집계
./run.sh all --dataset waterbirds
./run.sh status --dataset coco    # 진행 상황
./run.sh report                   # outputs/runs.csv, summary.csv, p1.json
```

- 끊겨도 **같은 명령을 다시 실행하면 이어서 돌아요.** 벤치마크 trainer가 매 epoch `resume.pt`를 남기고, 끝난 run은 건너뛰어요.
- 오래 걸리니 `nohup ./run.sh all --dataset coco > coco.out 2>&1 &`처럼 돌리는 걸 추천해요.

### 공유 GPU 예의
- 기본은 **한 번에 run 하나**예요. run을 시작하기 전마다 남은 VRAM을 확인해요(벤치마크 `scripts/gpu_capacity.py` 기준, 전역 여유 2.5 GiB).
- `--wait-pattern <문자열>`: 그 문자열과 맞는 프로세스(다른 사람 작업)가 돌고 있으면 다음 run을 시작하지 않고 기다려요. 이미 시작한 run은 멈추지 않아요.
- `--jobs 2`: 메모리가 되면 run 2개를 동시에 돌려요.

## 조건(arm)
| arm | teacher가 보는 패치 | 학생 손실 | 비고 |
|---|---|---|---|
| `mask98_ce1` | 98 (student attention top-k) | 0.5·CE + 0.5·KD | 벤치마크 MaskedKD 기준 run과 **같은 실행** |
| `mask98_ce0` | 98 | 0.5·KD | CE 항만 제거, **KD 가중치는 그대로** |
| `full_ce1` | 196 (전부) | 0.5·CE + 0.5·KD | |
| `full_ce0` | 196 | 0.5·KD | |
| `mask59_ce1/0` | 59 (~30%) | 위와 같음 | 선택 사항(`--arms`로 추가) |

## 벤치마크와의 관계
- 벤치마크 [reference 브랜치](https://github.com/jeehoo0507/kshs-aimlab-benchmarks/tree/codex/maskedkd-reference-runs)를 `external/`에 **커밋 `e1c39e7`로 고정**해 받아서, trainer(`reference.engine.train_run`)를 **그대로** 불러 써요.
- 우리가 바꾸는 건 [`probe/runner.py`](probe/runner.py)의 세 가지뿐이에요.
  1. 저장 위치: `outputs/<dataset>/<arm>/seed_<s>`
  2. `keep_patches`: 98 또는 196
  3. CE를 끈 조건에서 학생 CE 항을 0으로
- 데이터 분할, 전처리, 최적화, 스케줄, **검증셋 기준 체크포인트 선택**, test 평가, 마스크 진단, `result.json`은 전부 벤치마크 코드 그대로예요.
- 벤치마크 코드와 데이터는 이 저장소에 **포함하지 않아요.**
- `./run.sh report`가 `mask98_ce1` 평균을 벤치마크 결과표(`results/<dataset>/summary.csv`)와 비교해 줘요(재현 검사).
- teacher 해시까지 같게 비교하려면 연구실 기준 teacher 파일을 받아 `--teacher-ckpt path/to/best.pt`로 넘기면 돼요. 없으면 같은 규칙으로 teacher를 새로 학습해요. 이 경우 해시가 달라지지만, **우리 조건들끼리의 비교는 같은 teacher라 공정해요.**

## 예상 시간 (RTX A5000, 벤치마크 측정값 기준)
- 학생 run 1개: 약 1.2~1.4시간
- teacher: 약 0.4시간
- 데이터셋 하나에 4조건 × 3 seed = 12 run → 한 번에 하나씩이면 **약 16~17시간**. `--jobs 2`면 더 짧아요.

## 파이프라인 점검 (GPU 없이)
```bash
TORCH_INDEX=https://download.pytorch.org/whl/cpu ./run.sh setup
./run.sh smoke     # 가짜 데이터 + 작은 모델, CPU로 약 1분: teacher → 12 run → report
```

## 결과 파일
| 파일 | 내용 |
|---|---|
| `outputs/runs.csv` | run당 한 줄. 벤치마크 `runs.csv`와 같은 열에 `ce_in_loss`를 더함(method = arm) |
| `outputs/summary.csv` | 조건별 3-seed 평균·표준편차(검증 기준 체크포인트 / 마지막 epoch) |
| `outputs/p1.json` | seed별로 짝지은 gap_1, gap_0, ΔCE와 판정([PREREGISTRATION.md](PREREGISTRATION.md)), 기준 재현 검사 |

## License
MIT (이 저장소의 코드만). 벤치마크와 데이터는 각자의 조건을 따라요.
