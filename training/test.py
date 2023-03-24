# -*- coding: utf-8 -*-
# 💾⚙️🔮

__author__ = "Tom Potter"
__email__ = "tom.potter@bbc.co.uk"

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from simpletransformers.ner import NERModel
from rpunct.punctuate import VALID_LABELS

# sns.set_theme(style="darkgrid")
# sns.set(rc={'figure.figsize':(10, 7), 'figure.dpi':100, 'savefig.dpi':100})
plt.style.use('two-panel')

DATA_PATH = './training/datasets/'
RESULTS_PATH = './tests/'


def e2e_test(models, data_source='news-transcripts', use_cuda=True, output_file='model_performance.png'):
    """
    Testing model performance after full training process has been completed using prepared test dataset.
    """
    all_metrics = []
    count = 1

    for model_path in models:
        # Load each (fully trained) model to be tested
        model = NERModel(
            "bert",
            model_path,
            labels=VALID_LABELS,
            use_cuda=use_cuda,
            args={"max_seq_length": 512}
        )
        print(f"\n> Model {count}: {model_path}", end='\n\n')
        count += 1

        # Evaluate it on the test dataset to give precision/recall metrics
        test_data_txt = os.path.join(DATA_PATH, data_source, 'rpunct_test_set.txt')
        metrics, _, _ = model.eval_model(test_data_txt, output_dir=RESULTS_PATH)

        all_metrics.append(metrics)
        print(f"\n\t* Results: {metrics}")

    compare_models(all_metrics, models, out_png=output_file, data_type=data_source)

    print("\n> Model testing complete", end='\n\n')


def compare_models(results, model_locations, out_png='model_performance.png', data_type='news'):
    """
    Plotting function to visually compare the precision/recall of each tested model through a bar chart.
    """
    plot_path = os.path.join(RESULTS_PATH, out_png)  # output file to plot to
    df = pd.DataFrame(columns = ['Metrics', 'Results', 'Model'])  # dataframe for storing metrics to be plotted

    # Construct dataframe enumerating performance metrics for all models being compared
    count = 0
    for result in results:
        # Build individual df for a single model
        model_name_loc = model_locations[count].rfind('/')
        if model_name_loc == -1:
            model_name = model_locations[count]
        else:
            model_name = model_locations[count][model_name_loc + 1:]

        df2 = pd.DataFrame({
            'Metrics': [key.replace('_', ' ').capitalize() for key in result.keys()],
            'Results': result.values(),
            'Model': model_name
        })

        # Add this df to the global metrics df
        df = pd.concat([df, df2])
        count += 1

    # Plot & save single bar chart enumerating all model results
    fig, ax = plt.subplots(1, 1)
    sns.barplot(ax=ax, x='Metrics', y='Results', hue='Model', data=df)
    ax.set(title=f"Test Performance of Optimised Models ({data_type} data)")

    # set highest result for each metric to be more opaque than the rest
    count = 0
    for model in ax.containers:
        for bar in model:
            metric = df['Metrics'][count]
            result = df['Results'][count]
            metric_max = max(df[df['Metrics'] == metric]['Results'])

            if result == metric_max:
                bar.set_edgecolor('000')
                bar.set_zorder(2)
            else:
                bar.set_alpha(0.6)

            count += 1

    # plot details
    plt.xticks(rotation=90)
    plt.ylim(min(df['Results'].replace(0.0, 1.0)) - 0.05, max(df['Results']) + 0.05)
    plt.legend(loc=1)
    plt.tight_layout()

    fig.savefig(plot_path)
    print(f"\n> Performance comparison saved to: {plot_path}")


if __name__ == "__main__":
    data = 'news-transcripts'
    models = ['outputs/best_model', 'felflare/bert-restore-punctuation']

    e2e_test(models, data_type=data, use_cuda=False)
