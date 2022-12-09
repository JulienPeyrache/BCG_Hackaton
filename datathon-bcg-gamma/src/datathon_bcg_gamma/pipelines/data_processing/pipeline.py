"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import add_days,add_days_input, replace_na, attach_meteo, add_jours_f,prepare_data_for_model_taux,\
    prepare_data_for_model_debit, prepare_input_for_model, train_model, evaluate_models, evaluate_models_output,\
    reconstruct_output, concat_final, test_output, dump_csv

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func = add_days,
            inputs="data_champs_elysee",
            outputs="champs_elysee_with_days"
        ),
        node(
            func=replace_na,
            inputs="champs_elysee_with_days",
            outputs = "champs_elysee_with_days_na_filled"
        ),
        node(
            func=attach_meteo,
            inputs=['data_meteo','champs_elysee_with_days_na_filled'],
            outputs='df_champs_elysee_days_meteo'
        ),
        node(
            func=add_jours_f,
            inputs=['data_jours_feries','df_champs_elysee_days_meteo'],
            outputs='df_champs_elysee_days_meteo_bank'
        ),
        node(
            func=add_days_input,
            inputs="data_input_champs_elysee",
            outputs="champs_elysee_input_with_days"
        ),
        node(
            func=replace_na,
            inputs="champs_elysee_input_with_days",
            outputs="champs_elysee_input_with_days_na_filled"
        ),
        node(
            func=attach_meteo,
            inputs=['data_meteo_input', 'champs_elysee_input_with_days_na_filled'],
            outputs='df_champs_elysee_input_days_meteo'
        ),
        node(
            func=add_jours_f,
            inputs=['data_jours_feries', 'df_champs_elysee_input_days_meteo'],
            outputs='df_champs_elysee_input_days_meteo_bank'
        ),
        node(
            func = add_days,
            inputs="data_convention",
            outputs="convention_with_days"
        ),
        node(
            func=replace_na,
            inputs="convention_with_days",
            outputs = "convention_with_days_na_filled"
        ),
        node(
            func=attach_meteo,
            inputs=['data_meteo','convention_with_days_na_filled'],
            outputs='df_convention_days_meteo'
        ),
        node(
            func=add_jours_f,
            inputs=['data_jours_feries','df_convention_days_meteo'],
            outputs='df_convention_days_meteo_bank'
        ),
        node(
            func=add_days_input,
            inputs="data_input_conv",
            outputs="convention_input_with_days"
        ),
        node(
            func=replace_na,
            inputs="convention_input_with_days",
            outputs="convention_input_with_days_na_filled"
        ),
        node(
            func=attach_meteo,
            inputs=['data_meteo_input', 'convention_input_with_days_na_filled'],
            outputs='df_convention_input_days_meteo'
        ),
        node(
            func=add_jours_f,
            inputs=['data_jours_feries', 'df_convention_input_days_meteo'],
            outputs='df_convention_input_days_meteo_bank'
        ),
        node(
            func = add_days,
            inputs="data_peres",
            outputs="peres_with_days"
        ),
        node(
            func=replace_na,
            inputs="peres_with_days",
            outputs = "peres_with_days_na_filled"
        ),
        node(
            func=attach_meteo,
            inputs=['data_meteo','peres_with_days_na_filled'],
            outputs='df_peres_days_meteo'
        ),
        node(
            func=add_jours_f,
            inputs=['data_jours_feries','df_peres_days_meteo'],
            outputs='df_peres_days_meteo_bank'
        ),
        node(
            func=add_days_input,
            inputs="data_input_sts",
            outputs="peres_input_with_days"
        ),
        node(
            func=replace_na,
            inputs="peres_input_with_days",
            outputs="peres_input_with_days_na_filled"
        ),
        node(
            func=attach_meteo,
            inputs=['data_meteo_input', 'peres_input_with_days_na_filled'],
            outputs='df_peres_input_days_meteo'
        ),
        node(
            func=add_jours_f,
            inputs=['data_jours_feries', 'df_peres_input_days_meteo'],
            outputs='df_peres_input_days_meteo_bank'
        ),
        #Champs
        node(
            func=prepare_data_for_model_taux,
            inputs=['df_champs_elysee_days_meteo_bank','params:data_processing.past_value','params:data_processing.output_value'],
            outputs=['X_train_preprocessed_taux_che', 'X_test_preprocessed_taux_che', 'y_train_normalize_taux_che', 'y_train_taux_che', 'y_test_taux_che']
        ),
        node(
            func=prepare_data_for_model_debit,
            inputs=['df_champs_elysee_days_meteo_bank','params:data_processing.past_value','params:data_processing.output_value'],
            outputs=['X_train_preprocessed_debit_che', 'X_test_preprocessed_debit_che', 'y_train_normalize_debit_che', 'y_train_debit_che', 'y_test_debit_che']
        ),
        #Champs input
        node(
            func=prepare_input_for_model,
            inputs=['df_champs_elysee_input_days_meteo_bank', 'params:data_processing.past_value'],
            outputs=['X_train_preprocessed_taux_che_inp','date_taux_che']
        ),
        node(
            func=prepare_input_for_model,
            inputs=['df_champs_elysee_input_days_meteo_bank', 'params:data_processing.past_value'],
            outputs=['X_train_preprocessed_debit_che_inp','date_debit_che']
        ),
        # Convention input
        node(
            func=prepare_input_for_model,
            inputs=['df_convention_input_days_meteo_bank','params:data_processing.past_value'],
            outputs=['X_train_preprocessed_taux_conv_inp','date_taux_conv']
        ),
        node(
            func=prepare_input_for_model,
            inputs=['df_convention_input_days_meteo_bank','params:data_processing.past_value'],
            outputs=['X_train_preprocessed_debit_conv_inp','date_debit_conv']
        ),
        # Convention
        node(
            func=prepare_data_for_model_taux,
            inputs=['df_convention_days_meteo_bank', 'params:data_processing.past_value',
                    'params:data_processing.output_value'],
            outputs=['X_train_preprocessed_taux_conv', 'X_test_preprocessed_taux_conv', 'y_train_normalize_taux_conv',
                     'y_train_taux_conv', 'y_test_taux_conv']
        ),
        node(
            func=prepare_data_for_model_debit,
            inputs=['df_convention_days_meteo_bank', 'params:data_processing.past_value',
                    'params:data_processing.output_value'],
            outputs=['X_train_preprocessed_debit_conv', 'X_test_preprocessed_debit_conv',
                     'y_train_normalize_debit_conv', 'y_train_debit_conv', 'y_test_debit_conv']
        ),

        #St pere
        node(
            func=prepare_data_for_model_taux,
            inputs=['df_peres_days_meteo_bank','params:data_processing.past_value','params:data_processing.output_value'],
            outputs=['X_train_preprocessed_taux_peres', 'X_test_preprocessed_taux_peres', 'y_train_normalize_taux_peres', 'y_train_taux_peres', 'y_test_taux_peres']
        ),
        node(
            func=prepare_data_for_model_debit,
            inputs=['df_peres_days_meteo_bank','params:data_processing.past_value','params:data_processing.output_value'],
            outputs=['X_train_preprocessed_debit_peres', 'X_test_preprocessed_debit_peres', 'y_train_normalize_debit_peres', 'y_train_debit_peres', 'y_test_debit_peres']
        ),

        # St pere input
        node(
            func=prepare_input_for_model,
            inputs=['df_peres_input_days_meteo_bank', 'params:data_processing.past_value'],
            outputs=['X_train_preprocessed_taux_peres_inp','date_taux_peres']
        ),
        node(
            func=prepare_input_for_model,
            inputs=['df_peres_input_days_meteo_bank', 'params:data_processing.past_value'],
            outputs=['X_train_preprocessed_debit_peres_inp','date_debit_peres']
        ),

        #train champs
        # node(
        #     func=train_model,
        #     inputs=['X_train_preprocessed_taux_che','y_train_normalize_taux_che','params:data_processing.params_model'],
        #     outputs='model_taux_che'
        # ),
        # node(
        #     func=train_model,
        #     inputs=['X_train_preprocessed_debit_che','y_train_normalize_debit_che','params:data_processing.params_model'],
        #     outputs='model_debit_che'
        # ),
        # #train convention
        # node(
        #     func=train_model,
        #     inputs=['X_train_preprocessed_taux_conv','y_train_normalize_taux_conv','params:data_processing.params_model'],
        #     outputs='model_taux_conv'
        # ),
        # node(
        #     func=train_model,
        #     inputs=['X_train_preprocessed_debit_conv','y_train_normalize_debit_conv','params:data_processing.params_model'],
        #     outputs='model_debit_conv'
        # ),
        # #train st peres
        # node(
        #     func=train_model,
        #     inputs=['X_train_preprocessed_taux_peres','y_train_normalize_taux_peres','params:data_processing.params_model'],
        #     outputs='model_taux_peres'
        # ),
        # node(
        #     func=train_model,
        #     inputs=['X_train_preprocessed_debit_peres','y_train_normalize_debit_peres','params:data_processing.params_model'],
        #     outputs='model_debit_peres'
        # ),

        # evaluation modèle
        node(
            func=evaluate_models,
            inputs=['model_taux_che','X_test_preprocessed_taux_che','y_test_taux_che','y_train_taux_che' ],
            outputs='RMSE_taux_che'
        ),
        node(
            func=evaluate_models,
            inputs=['model_debit_che','X_test_preprocessed_debit_che','y_test_debit_che','y_train_debit_che' ],
            outputs='RMSE_debit_che'
        ),
        node(
            func=evaluate_models,
            inputs=['model_taux_conv','X_test_preprocessed_taux_conv','y_test_taux_conv','y_train_taux_conv' ],
            outputs='RMSE_taux_conv'
        ),
        node(
            func=evaluate_models,
            inputs=['model_debit_conv','X_test_preprocessed_debit_conv','y_test_debit_conv','y_train_debit_conv' ],
            outputs='RMSE_debit_conv'
        ),
        node(
            func=evaluate_models,
            inputs=['model_taux_peres','X_test_preprocessed_taux_peres','y_test_taux_peres','y_train_taux_peres' ],
            outputs='RMSE_taux_peres'
        ),
        node(
            func=evaluate_models,
            inputs=['model_debit_peres','X_test_preprocessed_debit_peres','y_test_debit_peres','y_train_debit_peres' ],
            outputs='RMSE_debit_peres'
            ),

        # evaluation output

        node(
            func=evaluate_models_output,
            inputs=['model_taux_che', 'X_train_preprocessed_taux_che_inp', 'y_train_taux_che'],
            outputs='output_taux_che'
        ),
        node(
            func=evaluate_models_output,
            inputs=['model_debit_che', 'X_train_preprocessed_debit_che_inp', 'y_train_debit_che'],
            outputs='output_debit_che'
        ),
        node(
            func=evaluate_models_output,
            inputs=['model_taux_conv', 'X_train_preprocessed_taux_conv_inp', 'y_train_taux_conv'],
            outputs='output_taux_conv'
        ),
        node(
            func=evaluate_models_output,
            inputs=['model_debit_conv', 'X_train_preprocessed_debit_conv_inp', 'y_train_debit_conv'],
            outputs='output_debit_conv'
        ),
        node(
            func=evaluate_models_output,
            inputs=['model_taux_peres', 'X_train_preprocessed_taux_peres_inp', 'y_train_taux_peres'],
            outputs='output_taux_peres'
        ),
        node(
            func=evaluate_models_output,
            inputs=['model_debit_peres', 'X_train_preprocessed_debit_peres_inp',
                    'y_train_debit_peres'],
            outputs='output_debit_peres'
        ),

        # reconstruct output
        node(
            func=reconstruct_output,
            inputs=['output_taux_che', 'output_debit_che', 'date_taux_che', 'params:data_processing.arc_che'],
            outputs='output_che'
        ),
        node(
            func=reconstruct_output,
            inputs=['output_taux_conv', 'output_debit_conv', 'date_taux_conv', 'params:data_processing.arc_conv'],
            outputs='output_conv'
        ),
        node(
            func=reconstruct_output,
            inputs=['output_taux_peres', 'output_debit_peres', 'date_taux_peres', 'params:data_processing.arc_peres'],
            outputs='output_peres'
        ),
        ## réunir tout en un

        node(
            func=concat_final,
            inputs=['output_peres', 'output_che', 'output_conv'],
            outputs='output_final'
        ),

        ## Test output

        node(
            func=test_output,
            inputs='output_final',
            outputs=None
        ),

        node(
            func=dump_csv,
            inputs='output_final',
            outputs='output_bcg'
        ),

    ])

