"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import add_days, attach_meteo, add_jours_f

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func = add_days,
            inputs="data_champs_elysee",
            outputs="champs_elysee_with_days"
        ),
        node(
            func=attach_meteo,
            inputs=['data_meteo','champs_elysee_with_days'],
            outputs='df_champs_elysee_days_meteo'
        ),
        node(
            func=add_jours_f,
            inputs=['data_jours_feries','df_champs_elysee_days_meteo'],
            outputs='df_champs_elysee_days_meteo_bank'
        )
    ])
