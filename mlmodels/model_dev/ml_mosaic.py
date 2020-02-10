
"""



pip install git+https://github.com/herilalaina/mosaic_ml


https://github.com/herilalaina/mosaic


https://www.ijcai.org/proceedings/2019/0457.pdf



"""


X_train, y_train, X_test, y_test, cat = load_task(6)

autoML = AutoML(time_budget=120,
                time_limit_for_evaluation=100,
                memory_limit=3024,
                seed=1,
                scoring_func="balanced_accuracy",
                exec_dir="execution_dir"
                )

best_config, best_score = autoML.fit(X_train, y_train, X_test, y_test, categorical_features=cat)
print(autoML.get_run_history())







