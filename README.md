# LMR-CL : Learning Modality-Fused Representations with Contrastive Loss for Multimodal Emotion Recognition 

## ❗️ Project Summary

---

1. **진행 기간:** 2023.02 ~ 2023.07
2. **역할:** 프로젝트 리더, 텍스트 특징 추출, Contrative Loss 수정 및 모델링
3. **기술 스택:**  **`Pytorch`**, **`HuggingFace`**
4. **결과 및 성과:** 
    - 특허 출원: 멀티모달 데이터의 융합을 기반으로 감정을 인식하기 위한 장치 및 방법
    *(출원 번호: 10-2023-0160075)*
    - ETRI 휴먼이해 인공지능 논문경진대회 과학기술정보통신부장관상 수상 [[🏅]](https://drive.google.com/file/d/1nsln5z21XBjtUogHpP5YfbaCQmJUwiBA/view)
    - 논문 게재 [[📄]](https://drive.google.com/file/d/1rNUtc2rlhUsqbbH1JOBMWbzLYgsePHfj/view)
5. **주요 내용:** 두 발화자의 자유 발화 과정에서의 텍스트, 음성, 생체신호로 구성된 멀티모달 감정 데이터 셋을 이용하여 중립을 포함한 7가지 감정을 분류하는 딥러닝 기반 감정 인식 모델을 개발하였습니다.
Multi-modal 데이터의 Alignment를 위해 배치 내 다른 샘플들 사이의 같은 모달리티 간 관계를 고려하는 Intra-modal Contrastive Loss와 다른 모달리티 간 관계를 파악하기 위한 Inter-modal Contrastive Loss를 적용한 방법을 제안하여 공모전에서 가장 높은 macro F1-score를 달성하였습니다.
6. **보도 자료:** [**[news1]](https://www.etnews.com/20230621000124) [[news2]](http://biz.heraldcorp.com/view.php?ud=20230621000315) [[news3]](https://www.gttkorea.com/news/articleView.html?idxno=5685)**

---

The "Han Hye-sung" team, composed of students from the Graduate School of Software Convergence at Kyung Hee University, won the Minister of Science and ICT Award at the **'2nd ETRI Human Understanding AI Paper Competition'** held by the Electronics and Telecommunications Research Institute (ETRI). 

(link : https://www.gttkorea.com/news/articleView.html?idxno=5685)

paper link : https://www.dbpia.co.kr/pdf/pdfView.do?nodeId=NODE11488658&googleIPSandBox=false&mark=0&ipRange=false&b2cLoginYN=false&isPDFSizeAllowed=true&accessgl=Y&language=ko_KR&hasTopBanner=true

This project implements a multimodal emotion recognition system based on the KEMDy20 dataset using text (BERT-based), audio (Wav2Vec2-based), and biosignal data. The project is organized in a modular fashion with separate components for data preprocessing, dataset creation, data loading, model definition, training, and evaluation.

---

## Repository Structure

```
LMR-CL/
├── Data/                         # Raw and processed input data (KEMDy20_v1_1)
├── checkpoints/                  # Saved model checkpoints
├── config/                       # Configuration and hyperparameter definitions
│   └── config.py                 # CLI arguments and default settings
├── data/                         # Preprocessed datasets and metadata
├── dataloaders/                  # PyTorch Dataset & DataLoader implementations
├── models/                       # Model architectures (LMR, etc.)
├── previous_code_files/          # Legacy or reference code
├── solvers/                      # Training and evaluation logic
│   ├── solver_bert.py            # Text modality solver
│   ├── solver_wav.py             # Audio modality solver
│   └── solver_wav_finetuning.py  # Audio fine-tuning solver
├── train/                        # Entry-point training scripts
│   ├── train_bert.py             # Train text subnetwork
│   ├── train_wav.py              # Train audio subnetwork
│   └── train_wav_finetuning.py   # Fine-tune audio network
├── utils/                        # Utility functions and helpers
├── environment.yml               # Conda environment specification
├── LICENSE                       # MIT License
└── README.md                     # This file
```

## Prerequisites

- **Conda** (Miniconda or Anaconda)
- **Python** 3.7+
- **CUDA** 10.1 (for GPU-enabled training)

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/hannn0403/LMR-CL.git
   cd LMR-CL
   ```
2. Create the Conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate lmr_cl
   ```
3. Install any additional dependencies (if needed):
   ```bash
   pip install -r requirements.txt  # if you add one
   ```

## Data Preparation

1. Download the **KEMDy20_v1_1** dataset and extract it into the `Data/` directory:
   ```bash
   Data/
   └── KEMDy20_v1_1/
       ├── audio/
       ├── text/
       └── biosignal/
   ```
2. The `config/config.py` script will look for this path by default; adjust `--data_dir` if your directory differs.

## Configuration

All hyperparameters and experiment settings can be modified via command-line arguments in `config/config.py`. Key options include:

```bash
--mode [train|eval]         # Run mode
--data kemdy20             # Dataset name
--use_bert True            # Enable text modality
--use_cmd_sim True         # Use contrastive similarity loss
--batch_size 256
--n_epoch 3
--learning_rate 1e-4
--optimizer Adam
--activation relu
--diff_weight 0.1          # Contrastive loss weight
--recon_weight 0.3         # Reconstruction loss weight
```  

To view all options:
```bash
python -c "from config.config import get_config; get_config(parse=True)"
```

## Training

### Text-only Subnetwork

```bash
python train/train_bert.py \
  --mode train \
  --data kemdy20 \
  --use_bert True \
  --name bert_experiment
```

### Audio-only Subnetwork

```bash
python train/train_wav.py \
  --mode train \
  --data kemdy20 \
  --name wav_experiment
```

### Audio Fine-Tuning

```bash
python train/train_wav_finetuning.py \
  --mode train \
  --data kemdy20 \
  --name wav_ft_experiment
```

Checkpoints will be saved under `checkpoints/<experiment_name>/`.

## Evaluation

Use the corresponding solver to evaluate on the validation/test split:

```bash
python solvers/solver_bert.py \
  --mode eval \
  --load_checkpoint checkpoints/bert_experiment/best_model.pth
```

Replace `solver_bert.py` with `solver_wav.py` or `solver_wav_finetuning.py` as needed.

## Logging & Visualization

- Training logs are printed to console; integrate TensorBoard by adapting `SummaryWriter` calls in solvers.
- Use your preferred visualization tools to plot loss curves and modality alignment metrics.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## References

- **Paper**: Han et al., *Learning Modality-Fused Representations with Contrastive Loss for Multimodal Emotion Recognition*.
- **Competition**: Minister of Science and ICT Award, 2nd ETRI Human Understanding AI Paper Competition.
- **Dataset**: [KEMDy20](https://github.com/kwonmheo/KEMDy20)  

---

