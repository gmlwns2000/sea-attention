mkdir -p submission
git checkout-index -a -f --prefix=./submission/
rm submission/build_docker.sh
rm submission/src/main/export_wandb.py
rm submission/src/poc/neko/test_causal_conv.ipynb
mkdir -p submission/plots
cp -i -r plots/* submission/plots
echo done