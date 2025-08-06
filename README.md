# trippy-chain-of-thoughts



dependencies:
conda create -n trippy-cot python=3.11 -c conda-forge -y
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install "transformers>=4.42" \
            "datasets>=2.19" \
            "trl>=0.8.0"        \
            bitsandbytes        \
            peft                \
            accelerate          \
            sentencepiece