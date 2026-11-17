"""Prospective out-of-year prediction. No target section appears in model training."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None

CAT = ["course","group","faculty","cohort","ep","project"]
NUM = ["credits","semester","year_start","prior_count","prior_mean","prior_std",
       "prior_group_mean","prior_group_count","prior_credits"]


def _feature_table(df:pd.DataFrame)->pd.DataFrame:
    x=df[CAT+NUM].copy()
    for col in CAT:
        x[col]=x[col].fillna("UNKNOWN").astype(str)
    for col in NUM:
        x[col]=pd.to_numeric(x[col],errors="coerce")
    return x


def _ridge():
    preprocess=ColumnTransformer([
        ("cat",OneHotEncoder(handle_unknown="ignore",min_frequency=3), CAT),
        ("num",Pipeline([("imp",SimpleImputer(strategy="constant",fill_value=0)),
                         ("std",StandardScaler())]),NUM)
    ])
    return Pipeline([("features",preprocess),("reg",Ridge(alpha=30.0,solver="lsqr"))])


def out_of_time_predictions(df:pd.DataFrame)->tuple[pd.DataFrame,dict]:
    out=df.copy()
    out["pred_ridge"]=np.nan
    out["pred_catboost"]=np.nan
    out["expected"]=np.nan
    out["model_name"]="NOT_AVAILABLE"
    observed=out.score.notna() & out.year_start.notna()
    years=sorted(out.loc[observed,"year_start"].astype(int).unique().tolist())
    x=_feature_table(out)
    results=[]
    accumulated=[]
    for year in years:
        training=observed & (out.year_start<year)
        testing=observed & (out.year_start==year)
        if training.sum()<200 or testing.sum()==0:
            results.append({"year":int(year),"train":int(training.sum()),"test":int(testing.sum()),"chosen":"NOT_AVAILABLE"})
            continue
        ridge=_ridge()
        ridge.fit(x.loc[training],out.loc[training,"score"])
        pred_ridge=np.clip(ridge.predict(x.loc[testing]),0,10)
        out.loc[testing,"pred_ridge"]=pred_ridge
        pred_cat=None
        if CatBoostRegressor is not None:
            # CatBoost accepts category strings + numeric NaN. Entire holdout year is excluded.
            model=CatBoostRegressor(iterations=140,depth=5,learning_rate=0.075,
                l2_leaf_reg=5,loss_function="RMSE",random_seed=2026,
                verbose=False,thread_count=2,allow_writing_files=False)
            model.fit(x.loc[training],out.loc[training,"score"],cat_features=CAT)
            pred_cat=np.clip(model.predict(x.loc[testing]),0,10)
            out.loc[testing,"pred_catboost"]=pred_cat
        # Select from PREVIOUS heldout years only (not from today's labels).
        if not accumulated or pred_cat is None:
            chosen="RIDGE"
        else:
            prior_cat=[a for a in accumulated if a[1] is not None]
            if not prior_cat:
                chosen="RIDGE"
            else:
                ridge_err=np.mean([a[0] for a in prior_cat])
                cat_err=np.mean([a[1] for a in prior_cat])
                chosen="CATBOOST" if cat_err < ridge_err*0.99 else "RIDGE"
        selected=pred_cat if chosen=="CATBOOST" and pred_cat is not None else pred_ridge
        out.loc[testing,"expected"]=selected
        out.loc[testing,"model_name"]=chosen
        actual=out.loc[testing,"score"].to_numpy()
        ridge_mae=float(mean_absolute_error(actual,pred_ridge))
        cat_mae=float(mean_absolute_error(actual,pred_cat)) if pred_cat is not None else None
        accumulated.append((ridge_mae,cat_mae))
        results.append({"year":int(year),"train":int(training.sum()),"test":int(testing.sum()),
            "chosen":chosen,"ridge_mae":round(ridge_mae,4),
            "catboost_mae":round(cat_mae,4) if cat_mae is not None else None,
            "selected_mae":round(float(mean_absolute_error(actual,selected)),4)})
    valid=out.score.notna() & out.expected.notna()
    out["residual"]=out.score-out.expected
    metrics={"per_year":results,"available":int(valid.sum()),
             "total":int(out.score.notna().sum()),
             "selected_mae":round(float(mean_absolute_error(out.loc[valid,"score"],out.loc[valid,"expected"])),4) if valid.any() else None,
             "selected_rmse":round(float(np.sqrt(mean_squared_error(out.loc[valid,"score"],out.loc[valid,"expected"]))),4) if valid.any() else None,
             "caveat":"Dự báo năm Y chỉ dùng điểm những năm trước Y; năm đầu thiếu lịch sử được đánh dấu chưa đủ bằng chứng dự báo."}
    return out,metrics
