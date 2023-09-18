mkdir -p submission

# checkout from current git branch
git checkout-index -a -f --prefix=./submission/
# remove private sensitive data
rm submission/build_docker.sh
rm submission/src/main/export_wandb.py
rm submission/src/poc/neko/test_causal_conv.ipynb

# copy plots and evaluation results, removes OPT visualization due to size limit
mkdir -p submission/plots
cp -rf plots/* submission/plots
rm -rf submission/plots/visualize_opt
rm -rf submission/plots/visualize_opt_backup
rm -rf submission/plots/main/figure_visualization_opt/wikitext2_0
rm -rf submission/plots/main/figure_visualization_opt/wikitext2_1
rm -rf submission/plots/main/figure_visualization_opt/wikitext2_2
rm -rf submission/plots/main/figure_visualization_opt/wikitext2_3
rm -rf submission/plots/main/figure_visualization_opt/*.png

# zip the submission
timestamp=$(date +'%m_%d_%Y')
zip -r "submission_${timestamp}.zip" submission/*

echo done