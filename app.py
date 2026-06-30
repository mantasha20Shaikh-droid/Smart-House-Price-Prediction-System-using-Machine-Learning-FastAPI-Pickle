from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import numpy as np
import pickle
from datetime import datetime

# FastAPI Application

app = FastAPI(
    title="Smart House Price Prediction System",
    version="1.0"
)

# Templates

templates = Jinja2Templates(directory="templates")

# Load Pickle Model

with open("house_price_model.pkl", "rb") as file:
    model = pickle.load(file)

# Home Page

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )

# Prediction

@app.post("/predict", response_class=HTMLResponse)
async def predict(

    request: Request,

    
    # Customer Information
    

    customer_name: str = Form(...),
    email: str = Form(...),
    phone: str = Form(...),

    
    # Property Information
    

    property_type: str = Form(...),
    location: str = Form(...),
    city: str = Form(...),

    bedrooms: int = Form(...),
    bathrooms: int = Form(...),

    area: float = Form(...),

    parking: str | None = Form(None),
    pool: str | None = Form(None),
    garden: str | None = Form(None),

    
    # Machine Learning Inputs
    

    MedInc: float = Form(...),
    HouseAge: float = Form(...),
    AveRooms: float = Form(...),
    AveBedrms: float = Form(...),
    Population: float = Form(...),
    AveOccup: float = Form(...),
    Latitude: float = Form(...),
    Longitude: float = Form(...)

):

   
    # Checkbox Conversion
    

    parking = parking is not None
    pool = pool is not None
    garden = garden is not None

    
    # Prepare Data
    

    data = np.array([[
        MedInc,
        HouseAge,
        AveRooms,
        AveBedrms,
        Population,
        AveOccup,
        Latitude,
        Longitude
    ]])

    
    # Predict Price
    

    prediction = model.predict(data)[0]

    # California Housing Target
    prediction = prediction * 100000

    prediction = round(prediction, 2)

    
    # Date & Time
    

    prediction_date = datetime.now().strftime("%d-%m-%Y")

    prediction_time = datetime.now().strftime("%I:%M %p")

    
    # Return HTML
   
    return templates.TemplateResponse(

        request=request,

        name="index.html",

        context={

            # Prediction

            "prediction": f"$ {prediction:,.2f}",

            # Customer

            "customer_name": customer_name,
            "email": email,
            "phone": phone,

            # Property

            "property_type": property_type,
            "location": location,
            "city": city,

            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "area": area,

            "parking": parking,
            "pool": pool,
            "garden": garden,

            # Date

            "prediction_date": prediction_date,
            "prediction_time": prediction_time,

            # Graph Values

            "graph_income": MedInc,
            "graph_rooms": AveRooms,
            "graph_population": Population,
            "graph_age": HouseAge

        }

    )
