import pandas as pd
import numpy as np
import plotnine as p9

from src.misc import DataReader, prep_latex_tab, average_rank

OUTPUT_DIR = 'assets/results/outputs/'

THEME = p9.theme_538(base_family='Palatino', base_size=14) + \
        p9.theme(plot_margin=.015,
                 panel_background=p9.element_rect(fill='white'),
                 plot_background=p9.element_rect(fill='white'),
                 legend_box_background=p9.element_rect(fill='white'),
                 strip_background=p9.element_rect(fill='white'),
                 legend_background=p9.element_rect(fill='white'),
                 axis_text_x=p9.element_text(size=15, angle=30),
                 axis_text_y=p9.element_text(size=15),
                 legend_title=p9.element_blank())

cpu_df = DataReader.read_all_cpu_time(round_to=0)

cpu_time = (cpu_df.groupby('Detector')['CPU_Time'].mean() / 60).round(2).reset_index()

detector_order = cpu_time.sort_values('CPU_Time', ascending=False)['Detector']

cpu_time['Detector'] = pd.Categorical(cpu_time['Detector'], categories=detector_order, ordered=True)

p_bar = (
        p9.ggplot(cpu_time, p9.aes(x='Detector', y='CPU_Time'))
        + p9.geom_bar(stat='identity', fill='steelblue', show_legend=False)
        + p9.coord_flip()
        + THEME
        + p9.theme(axis_text_x=p9.element_text(angle=0), )
        + p9.labs(
    title='',
    x='Detector',
    y='CPU Time (minutes)'
)
)

p_bar.save(f"{OUTPUT_DIR}/plot2_cpu.pdf", width=6, height=6)
