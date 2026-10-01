# kd-bench-probe

**정답 라벨(CE)이 teacher 마스킹의 손해를 가려 주는가?** [KSHS × AIM Lab 벤치마크](https://github.com/jeehoo0507/kshs-aimlab-benchmarks)에서 확인하는 실험 코드이다.

*Does ground-truth CE hide the cost of teacher-side masking (MaskedKD)? A probe on the KSHS × AIM Lab benchmark, using its reference trainer unchanged.*

## 실행
```bash
git clone https://github.com/NGC1788/kd-bench-probe
cd kd-bench-probe
./kdb all --dataset coco
./kdb all --dataset waterbirds
```
- `./kdb`는 세 줄짜리 실행 파일이다. uv로 **Python 3.11을 폴더 안(`.cache/python`)에** 받은 뒤 `uv run`을 불러온다. 서버 파이썬 버전(예: 3.14)과 상관없이 돈다.
- 처음 실행할 때 `uv.lock`에 고정된 패키지(torch 2.5.1+cu124 등, 벤치마크 기준 run과 같은 버전)를 폴더 안 `.venv`에 설치한다.
- 한 번 실행하면 데이터 준비 → teacher 학습(30 epoch) → 4조건 × 3 seed 학생 학습 → 결과 집계까지 **앞에서 차례로** 돌아요. 백그라운드 프로세스를 따로 띄우지 않고, 다른 사용자의 프로세스도 들여다보지 않는다.
- 중간에 끊겨도 **같은 명령을 다시 실행하면 이어서 해요.** 끝난 run은 건너뛴다.
- 진행 상황: `./kdb status --dataset coco`
- 결과만 다시 모으기: `./kdb report`
- teacher 포화 확인: `./kdb saturation` (아래 [teacher 포화 확인](#teacher-포화-확인))
- 검증 곡선 전체 비교(탐색용, 판정에는 쓰지 않음): `./kdb curves`. 선택된 체크포인트 한 점이 아니라 epoch 구간 평균으로 조건을 비교한다.

**서버에 필요한 것:** `uv`, `git`, `curl`, CUDA GPU. 파이썬은 `./kdb`가 폴더 안에 받는다.
**`uv run`을 직접 치지 말고 `./kdb`를 쓰세요.** 폴더 밖에 파이썬이 설치되지 않도록, 폴더 안에 파이썬이 없으면 `uv run`이 일부러 실패하게 해 두었다.
**클러스터에 작업 스케줄러가 있으면**(예: Slurm) 위 명령을 관리자가 안내한 작업 제출 방식으로 돌리면 된다. 명령 자체는 똑같다.

## 한 폴더에 전부
- 이 프로젝트가 만드는 건 **전부 이 폴더 안**에 생긴다: `.cache/python`(Python 3.11), `.venv`(패키지), `.cache`(uv 캐시, DeiT 가중치, CUDA 커널 캐시), `external`(벤치마크 코드), `outputs`(결과).
- **다 지우려면 폴더를 삭제하면 끝**이다: `rm -rf kd-bench-probe`

## 조건(arm)
| arm | teacher가 보는 패치 | 학생 손실 | 비고 |
|---|---|---|---|
| `mask98_ce1` | 98개 (student attention 상위) | 0.5·CE + 0.5·KD | 벤치마크 MaskedKD 기준 run과 **같은 실행** |
| `mask98_ce0` | 98개 | 0.5·KD | CE 항만 제거, **KD 가중치는 그대로** |
| `full_ce1` | 196개 (전부) | 0.5·CE + 0.5·KD | |
| `full_ce0` | 196개 | 0.5·KD | |
| `mask59_ce1/0` | 59개 (~30%) | 위와 같음 | 선택 사항(`--arms`로 추가) |

## 벤치마크와의 관계
- 벤치마크 [reference 브랜치](https://github.com/jeehoo0507/kshs-aimlab-benchmarks/tree/codex/maskedkd-reference-runs)를 `external/`에 **커밋 `e1c39e7`로 고정**해 받아서, trainer를 **수정 없이** 불러 쓴다.
- [`probe/runner.py`](probe/runner.py)가 바꾸는 건 세 가지뿐이다.
  1. 저장 위치
  2. `keep_patches`(98 또는 196)
  3. CE를 끈 조건에서 학생의 CE 항을 0으로
- 데이터 준비는 벤치마크 스크립트(`datasets/…/prepare.py`, `download.py`, `download_masks.py`)를 이 프로젝트의 파이썬 하나로 직접 실행한다. 별도 가상환경을 더 만들지 않는다.
- `report`가 `mask98_ce1` 평균을 벤치마크 결과표와 비교해서 **재현 검사**를 해 준다.
- 연구실 기준 teacher 파일이 있으면 `--teacher-ckpt path/to/best.pt`로 넘겨요. 그러면 teacher 해시까지 기준과 같아진다.

## 예상 시간 (RTX A5000, 벤치마크 측정값 기준)
- 학생 run 1개: 약 1.2~1.4시간
- teacher: 약 0.4시간
- **데이터셋 하나(12 run): 약 16~17시간**, 두 데이터셋 합쳐 약 1.5일

## 파이프라인 점검 (GPU 없이)
```bash
./kdb smoke    # 가짜 데이터 + 작은 모델, CPU로 약 1분: teacher → 12 run → report
```
(맥이나 리눅스 CPU에서는 CPU판 torch가 설치돼요.)

## 결과 파일
| 파일 | 내용 |
|---|---|
| `outputs/runs.csv` | run당 한 줄. 벤치마크 `runs.csv`와 같은 열에 `ce_in_loss`를 더함(method = arm) |
| `outputs/summary.csv` | 조건별 3-seed 평균·표준편차 |
| `outputs/p1.json` | seed별로 짝지은 gap_1, gap_0, ΔCE와 판정([PREREGISTRATION.md](PREREGISTRATION.md)), 기준 재현 검사 |

## teacher 포화 확인
```bash
./kdb saturation
```
- 끝난 run의 `history.json`만 읽는다. GPU를 쓰지 않고 학습도 하지 않는다.
- teacher 출력이 학습 이미지에서 label smoothing 라벨 q와 같으면, 학생의 마지막 epoch에서 `train CE − train KD`가 q의 엔트로피 H(q)와 같아진다.
  - KD = KL(q‖s) = CE_ls(s) − H(q)이기 때문이다.
- 그러면 학습셋에서 KD가 라벨 이상의 정보를 주지 못한 것이다. full, 마스킹, CE 켬/끔이 사실상 같은 손실을 최적화한 셈이다.
- H(q) 기준값: COCO(10클래스) 0.5003, Waterbirds(2클래스) 0.1985 (ε = 0.1).
- CE를 끈 조건은 CE가 0으로 기록되기 때문에 `_ce1` 조건만 확인한다.

## License
MIT (이 저장소의 코드만). 벤치마크와 데이터는 각자의 조건을 따른다.
