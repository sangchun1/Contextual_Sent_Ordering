# 문맥 기반 문장 순서 예측 프로젝트

이 프로젝트는 [Dacon 문맥 기반 문장 순서 예측 AI 경진대회](https://dacon.io/competitions/official/236489/overview/description)에 참가하여, 주어진 문서의 문장들을 올바른 순서로 배열하는 딥러닝 모델을 개발하는 것을 목표로 합니다. 

---

## 📌 프로젝트 개요

- **목표**: 주어진 4개의 문장을 올바른 순서로 재배열
- **접근법**:
  - Pairwise 방식: 문장 쌍 간 순서를 예측하여 전체 순서 결정
  - Sequence 방식: 전체 문장을 입력받아 순서를 직접 생성
  - Global 방식: 전체 문장을 동시에 고려하여 각 문장의 위치를 예측
- **사용 모델**:
  - `KLUE-RoBERTa` (Pairwise / Global)
  - `KLUE-BERT` (Pairwise / Global)
  - `KoElectra` (Pairwise / Global)
  - `T5` (Sequence)

---

## 📂 프로젝트 구조

```
📂 data/ # 원본 데이터 및 제출 파일
├── train.csv # 학습용 데이터 (7,350개 샘플)
├── test.csv # 테스트용 데이터 (1,780개 샘플)
├── sample_submission.csv # 제출 샘플 파일
└── submission.csv # 최종 제출 파일

📂 preprocess/ # 전처리 코드
├── preprocess.ipynb # 전처리 노트북
└── preprocessing.py # 전처리 모듈 (Pairwise/Sequence/Global)

📂 model test/ # 각 모델별 실험 코드
├── KLUE-BERT/
├── KLUE-RoBERTa/
├── KoElectra/
└── T5/

📂 model results/ # 최종 결과 및 평가지표 노트북
├── global/ # Global 방식 결과
├── pairwise/ # Pairwise 방식 결과
└── sequence/ # Sequence 방식 결과

📂 presentation/ # 발표 자료
├── 최종 보고서.docx
└── 최종 발표자료.pdf
```

---

## 📊 데이터셋

- **train.csv**: 학습용 데이터, 총 **7,350개 샘플**
- **test.csv**: 테스트용 데이터, 총 **1,780개 샘플**
- 각 샘플은 4개의 문장(`sentence_0` ~ `sentence_3`)으로 구성
- `train.csv`에는 정답 순서를 나타내는 `answer_0` ~ `answer_3` 컬럼 포함
- `test.csv`에는 정답이 없으며, 예측 결과를 `submission.csv`로 제출

---

## ⚙️ 전처리

- **Pairwise**: 각 샘플을 가능한 모든 문장쌍(24개)으로 분해 → 올바른 순서 여부(0/1) 라벨 부여  
- **Sequence**: 네 문장을 `[SEP]`으로 연결 → 정답 순서를 문자열(예: `"3 2 1 0"`)로 변환  
- **Global**: 네 문장을 `[SEP]`으로 연결 → 가능한 24개 순열을 클래스 레이블로 정의  
- **데이터 증강**: Sequence, Global 방식에서는 각 샘플에 대해 24가지 가능한 문장 순열을 모두 생성하여 학습 데이터 다양성을 확보  

---

## 📐 평가 지표

모델 성능은 단순 정확도 외에도 다양한 지표로 평가함:

- **Full Order Accuracy (FOA)**: 전체 문장 순서를 완벽히 맞춘 비율  
- **Sentence Accuracy (SA)**: 개별 문장이 올바른 위치에 배치된 비율  
- **Kendall’s Tau (KT)**: 예측 순서와 실제 순서 간 순위 일관성 측정  
- **Spearman’s Rho (SR)**: 순위 간 절대적 거리 차이를 반영  
- **Precision / Recall / F1**: 분류 모델 관점에서의 성능 평가  

---

## 🤖 모델링 전략

### 1. Pairwise Ranking
- **입력**: 문장 A, 문장 B → 두 문장의 순서를 비교  
- **학습**: 각 쌍에 대해 "A가 B보다 앞인가?"를 이진 분류  
- **예측**: 모든 쌍의 결과를 종합하여 최적의 순서 선택  

### 2. Sequence Prediction
- **입력**: `문장0 [SEP] 문장1 [SEP] 문장2 [SEP] 문장3`  
- **출력**: `"2 0 1 3"` 형태의 순서 문자열  
- **모델**: `paust/pko-t5-large`, `wisenut-nlp-team/KoT5-base`  

### 3. Global Classification
- **입력**: 네 문장을 하나의 시퀀스로 결합  
- **출력**: 가능한 24개 순열 중 하나를 분류  

---

## 📈 실험 결과

### Pairwise 방식
| 모델 | FOA | SA | KT | SR | F1 | Public | Private |
|------|-----|-----|-----|-----|----|--------|---------|
| BERT | 70.61 | 92.50 | 80.29 | 83.03 | 84.75 | 76.97 | 77.36 |
| KoELECTRA | 68.30 | 91.76 | 78.41 | 81.26 | 83.05 | 73.60 | 73.37 |
| RoBERTa | 59.39 | 88.93 | 72.11 | 75.73 | 78.03 | 78.31 | 77.92 |

### Sequence 방식
| 모델 | FOA | SA | KT | SR | Public | Private |
|------|-----|-----|-----|-----|--------|---------|
| T5 | 99.61 | 99.61 | 99.86 | 99.91 | 80.45 | 81.18 |

### Global 방식
| 모델 | FOA | SA | KT | SR | Public | Private |
|------|-----|-----|-----|-----|--------|---------|
| BERT | 99.66 | 99.83 | 99.87 | 99.92 | 79.76 | 78.43 |
| KoELECTRA | 99.62 | 99.81 | 99.84 | 99.88 | 78.76 | 79.61 |
| RoBERTa | 99.86 | 99.93 | 99.95 | 99.96 | 84.38 | 83.71 |

---

## 🔍 주요 발견사항

1. **Pairwise 한계**: 문장쌍 단위 학습은 국소적 관계에는 강점이 있으나 전체 순서를 복원하는 데 한계 존재  
2. **Sequence 과적합**: 내부 지표가 비정상적으로 높아 과적합 발생  
3. **Global 강점**: RoBERTa 기반 Global 방식이 가장 안정적이고 우수한 성능 달성 (Private 83.71%)  
4. **모델별 차이**: NSP를 제거한 RoBERTa가 BERT 대비 더 높은 일반화 성능  

---

## 📌 결론 및 향후 개선 방향

- **Global 방식**이 가장 효과적이며, 문장 순서 배열 문제에서 전체 문맥을 고려하는 것이 필수적임  
- **Pairwise**는 세밀한 문장 관계 파악에는 유리하나 전체 순서 복원에는 한계  
- **Sequence**는 직관적이지만 과적합 문제가 뚜렷함  
- 향후 개선 방향:
  1. 순서 혼동 영역(특히 2–3번 위치) 집중 학습  
  2. Pairwise + Global 결합 등 하이브리드 구조 탐색  
  3. 데이터 증강 및 다양한 permutation 기반 학습  
  4. 앙상블 기법을 통한 보완  
