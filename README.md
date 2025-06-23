# 문맥 기반 문장 순서 예측 프로젝트

이 프로젝트는 [Dacon 문맥 기반 문장 순서 예측 AI 경진대회](https://dacon.io/competitions/official/236489/overview/description)에 참가하여, 주어진 문서의 문장들을 올바른 순서로 배열하는 딥러닝 모델을 개발하는 것을 목표로 합니다. 

## 프로젝트 개요

- **목표**: 주어진 4개의 문장을 올바른 순서로 재배열
- **접근법**:
  - Pairwise 방식: 문장 쌍 간 순서를 예측하여 전체 순서 결정
  - Seq2Seq 방식: 전체 문장을 입력받아 순서를 직접 생성
  - Global 방식: 전체 문장을 동시에 고려하여 각 문장의 위치를 예측
- **사용 모델**:
  - `KLUE-RoBERTa` (Pairwise) - **88.90% 정확도**
  - `KLUE-BERT` (Pairwise) - **66.53% 정확도**
  - `KoElectra` (Pairwise) - **89.27% 정확도** (튜닝 후 **90.19%**)
  - `T5` (Seq2Seq) - **76.63% 정확도**
  - `Global RoBERTa` (Global) - **100% 정확도** (DACON 최고 성능)

## 프로젝트 구조

```
📂 data/                    # 원본 데이터 및 제출 파일
├── train.csv              # 학습용 데이터 (7,350개 샘플)
├── test.csv               # 테스트용 데이터 (1,780개 샘플)
├── sample_submission.csv  # 제출 샘플 파일
└── submission.csv         # 최종 제출 파일
📂 preprocess/             # 전처리 코드
├── preprocess.ipynb       # 전처리 노트북
└── preprocessing.py       # 전처리 모듈 (Pairwise/Seq2Seq)
📂 KLUE-RoBERTa/          # KLUE-RoBERTa 모델 실험
└── train_roberta.ipynb   # Pairwise 방식 학습

📂 KLUE-BERT/             # KLUE-BERT 모델 실험
└── train_bert.ipynb      # Pairwise 방식 학습

📂 KoElectra/             # KoElectra 모델 실험
└── train_koelectra.ipynb # Pairwise 방식 학습

📂 T5/                    # T5 모델 실험 (Seq2Seq)
├── ko_t5.ipynb          # 한국어 T5 모델
├── pko_t5.ipynb         # Polyglot-Ko T5 모델
└── grid_search_results/ # 하이퍼파라미터 탐색 결과
    └── pko_t5_results.csv

📂 global/               # 전역적 접근 방식
├── global_roberta.ipynb # 전역적 RoBERTa 모델
└── global_roberta(2).ipynb # 개선된 전역적 모델

📂 presentation/         # 발표 자료
├── 최종 보고서.docx     # 상세한 실험 보고서
└── 최종 발표자료.pdf    # 발표 슬라이드
```

## 데이터셋

- **train.csv**: 학습용 데이터, 총 **7,350개 샘플**
- **test.csv**: 테스트용 데이터, 총 **1,780개 샘플**
- 각 샘플은 4개의 문장(`sentence_0` ~ `sentence_3`)으로 구성
- `train.csv`에는 정답 순서를 나타내는 `answer_0` ~ `answer_3` 컬럼 포함
- `test.csv`에는 정답이 없으며, 예측 결과를 `submission.csv`로 제출

## 전처리

### Pairwise 모델용 전처리
- 각 샘플의 문장 4개에서 가능한 문장 순서쌍 12개를 생성
- 올바른 순서를 기준으로 label 부여 (예: A가 B보다 앞이면 `1`, 아니면 `0`)
- 함수: `preprocess_pairwise()`

### Seq2Seq 모델용 전처리
- 4개의 문장을 `[SEP]` 토큰으로 연결하여 하나의 입력으로 합침
- 정답 순서를 문자열로 생성 (예: `"2 0 1 3"`)
- 함수: `preprocess_seq2seq()`

### Global 모델용 전처리
- 4개의 문장을 하나의 시퀀스로 결합
- 각 문장의 절대적 위치(0~3)를 예측하는 구조

## 모델링 전략

### 1. Pairwise Ranking (KLUE-RoBERTa, KLUE-BERT, KoElectra)
- **입력**: 문장 A, 문장 B → 두 문장의 순서를 비교
- **학습**: 각 쌍에 대해 "A가 B보다 앞인가?"를 이진 분류
- **예측**: 12개 쌍의 예측을 바탕으로 가능한 문장 순서 조합 중 가장 일관된 조합 선택
- **장점**: 문장 간의 로컬 관계를 정밀하게 포착
- **단점**: 전체 논리적 흐름 반영의 한계, 계산량 부담

### 2. Seq2Seq Prediction (T5)
- **입력**: `문장0 [SEP] 문장1 [SEP] 문장2 [SEP] 문장3`
- **출력**: `"2 0 1 3"` 형태의 순서 문자열
- **모델**: 
  - `ko_t5`: 한국어 T5 모델
  - `pko_t5`: Polyglot-Ko T5 모델 (그리드 서치 완료)
- **장점**: 직관적인 구조, 자연어 처리와 유사
- **단점**: 긴 입력에 취약, 미묘한 순서 차이 구분 어려움

### 3. Global Approach (Global RoBERTa)
- **입력**: 4개 문장을 하나의 시퀀스로 결합
- **출력**: 각 문장의 절대적 위치(0~3) 예측
- **장점**: 전체 문맥 고려, 논리적 일관성 포착
- **단점**: 과적합 위험, 확정적 예측의 제약

## 실험 결과

### Pairwise 모델 성능 비교
| 모델 | 정확도 | 검증 손실 | 특징 |
|------|--------|-----------|------|
| KLUE-BERT | 66.53% | 0.6354 | 기본 BERT, 상대적으로 낮은 성능 |
| KLUE-RoBERTa | 88.90% | 0.4405 | NSP 없이 문장 수준 이해에 집중 |
| KoElectra | 89.27% | 0.3371 | Generator-Discriminator 방식, 최고 성능 |

### T5 모델 성능
- **paust/pko-t5-large**: 더 우수한 성능으로 최종 선택
- **wisenut-nlp-team/KoT5-base**: 경량화된 모델
- **최고 성능**: 76.63% (학습률 5e-05, 배치 크기 4, 에포크 5)

### Global 모델 성능
- **검증 정확도**: 거의 100% (과적합 현상)
- **테스트 정확도**: 84.27% (DACON 리더보드 최고 성능)
- **문제점**: 2-3번 문장 위치 혼동, 과적합

### 하이퍼파라미터 최적화 결과
#### KoElectra 튜닝 (Random Search)
- **최적 조합**:
  - Learning Rate: 1.1921e-05
  - Weight Decay: 0.0219
  - Warmup Steps: 550
  - Epochs: 8
  - Train Batch Size: 63
  - Scheduler Type: cosine
- **성능**: 89.43% 정확도, 0.2992 손실

## 모델 구조 개선

### KoElectra 개선 사항
1. **Hidden Layer 추가 및 확장**: 여러 개의 hidden layer 도입
2. **Multi-Head Attention 도입**: 문장쌍 간 다양한 관계 병렬 파악
3. **Layer Normalization 적용**: 학습 안정성 및 수렴 속도 개선
4. **결과**: 90.19% 정확도 달성

## 의존성 패키지

### Core ML and Deep Learning
```
torch>=1.12.0
transformers>=4.35.0
scikit-learn>=1.1.0
```

### Data Processing
```
pandas>=1.3.0
numpy>=1.21.0
```

### Progress and Utilities
```
tqdm>=4.64.0
```

### Visualization
```
matplotlib>=3.5.0
```

### Scientific Computing
```
scipy>=1.9.0
```

### Jupyter and Notebook Support
```
jupyter>=1.0.0
ipywidgets>=7.6.0
nbformat>=5.0.0
```

### 설치 방법
```bash
pip install -r requirements.txt
```

**참고**: `os`, `gc`, `json`, `itertools`, `shutil` 등은 Python 빌트인 모듈로 별도 설치가 불필요합니다.

## 평가 방법

- **정확도**: 정확한 순서를 맞춘 sample 비율 (Accuracy)
- **제출 형식**: `submission.csv` (`id`, `pred`) 형태
- **DACON 리더보드**: 84.27% 정확도로 최고 성능 달성

## 주요 성과

- **DACON 경진대회 최고 성능**: Global RoBERTa 모델로 84.27% 달성
- **다양한 접근법 구현**: Pairwise, Seq2Seq, Global 접근법 모두 구현
- **하이퍼파라미터 최적화**: Random Search를 통한 체계적 튜닝
- **모델 구조 개선**: KoElectra 모델의 구조적 개선으로 90.19% 달성
- **모듈화**: 전처리 코드를 별도 모듈로 분리하여 재사용성 향상

## 결론 및 향후 개선 방향

### 주요 발견사항
1. **Global 방식이 가장 효과적**: 전체 문맥을 고려한 접근이 우수한 성능
2. **KoElectra의 우수성**: 한국어 특화 모델이 문장 순서 예측에 효과적
3. **과적합 문제**: Global 모델에서 검증/테스트 성능 차이 발생

### 향후 개선 방향
1. **순서 혼동 영역 집중 학습**: 2-3번 문장 위치 혼동 해결
2. **하이브리드 구조**: Pairwise와 Global 방식의 결합
3. **데이터 증강**: 다양한 문장 조합 패턴 학습
4. **앙상블 기법**: 다양한 관점의 예측 통합
