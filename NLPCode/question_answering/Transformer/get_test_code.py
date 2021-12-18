if __name__ == "__main__":
    %%bash
    mkdir hugging-face-squad
    echo > hugging-face-squad/__init__.py
    cd hugging-face-squad
    wget 'https://raw.githubusercontent.com/huggingface/pytorch-transformers/master/examples/run_squad.py'
    wget 'https://raw.githubusercontent.com/huggingface/pytorch-transformers/master/examples/utils_squad.py'
    wget 'https://raw.githubusercontent.com/huggingface/pytorch-transformers/master/examples/utils_squad_evaluate.py'
    sed -i 's/utils_squad_evaluate/.utils_squad_evaluate/g' utils_squad.py
    sed -i 's/utils_squad/.utils_squad/g' run_squad.py