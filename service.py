
import bentoml 
from bentoml.io import JSON
from pydantic import BaseModel, StrictStr
import numpy as np

class WineQualityApplication(BaseModel):
    type: StrictStr 
    fixed_acidity:float 
    volatile_acidity:float 
    citric_acid:float 
    residual_sugar:float 
    chlorides:float 
    free_sulfur_dioxide:float 
    total_sulfur_dioxide:float 
    density:float 
    ph:float 
    sulphates:float 
    alcohol:float 


model_ref = bentoml.sklearn.get("wine_quality_randomforest:ahhbkxtaq26rig2k") 
dv = model_ref.custom_objects['DictVectorizer']

model_runner = model_ref.to_runner()

svc = bentoml.Service("wine_quality_classifier",runners=[model_runner]) 

@svc.api(input=JSON(pydantic_model=WineQualityApplication),output=JSON())
async def classify(wineq_application):
    application_data = wineq_application.dict()
    print(application_data)
    if application_data['type'] == 'red':
        application_data['type'] = 0
    else:
        application_data['type'] = 1
    vector = dv.transform(application_data)
    prediction = await model_runner.predict_proba.async_run(vector)
    result = prediction[0].round(2)
    classes = ['bad','moderate','good']
    
    print('---------------')
    print(result)
    print('---------------')
    out = {
        'Probabilities': {quality:result[i] for i,quality in enumerate(classes)},
        'Quality':classes[np.argmax(result)]
        }
    return out

