"""
Complete disease information database for all 38 plant disease classes
Provides specific symptoms, causes, and treatments for each disease
"""

DISEASE_DATABASE = {
    # Apple diseases
    "Apple___Apple_scab": {
        "symptoms": ["Olive-green to brown spots on leaves", "Scabby lesions on fruits", "Premature leaf drop", "Reduced fruit quality"],
        "causes": ["Fungal infection by Venturia inaequalis", "Cool wet spring weather", "Poor air circulation"],
        "treatments": {
            "chemical": ["Apply Captan or Myclobutanil fungicide", "Spray Mancozeb preventively"],
            "organic": ["Sulfur spray", "Neem oil application", "Remove infected leaves"],
            "prevention": ["Plant resistant varieties", "Prune for air circulation", "Remove fallen leaves"]
        }
    },
    
    "Apple___Black_rot": {
        "symptoms": ["Purple spots on leaves turning brown", "Sunken black lesions on fruits", "Cankers on branches"],
        "causes": ["Fungal infection by Botryosphaeria obtusa", "Warm humid weather", "Wounds on tree"],
        "treatments": {
            "chemical": ["Apply Captan or Thiophanate-methyl", "Prune infected branches"],
            "organic": ["Copper fungicide", "Remove mummified fruits", "Improve sanitation"],
            "prevention": ["Remove dead wood", "Avoid tree injuries", "Maintain tree vigor"]
        }
    },
    
    "Apple___Cedar_apple_rust": {
        "symptoms": ["Yellow-orange spots on upper leaf surface", "Tube-like structures on leaf undersides", "Premature defoliation"],
        "causes": ["Fungal infection requiring cedar and apple trees", "Spring rainfall", "Proximity to cedar trees"],
        "treatments": {
            "chemical": ["Apply Myclobutanil or Propiconazole", "Spray during spring"],
            "organic": ["Remove nearby cedar trees", "Sulfur spray", "Resistant varieties"],
            "prevention": ["Plant resistant apple varieties", "Remove cedar trees within 2 miles", "Fungicide sprays in spring"]
        }
    },
    
    "Apple___healthy": {
        "symptoms": ["Healthy green foliage", "No disease symptoms", "Normal fruit development"],
        "causes": ["Good cultural practices", "Disease-free management"],
        "treatments": {
            "chemical": [],
            "organic": [],
            "prevention": ["Continue monitoring", "Maintain tree health", "Practice good sanitation"]
        }
    },
    
    # Blueberry
    "Blueberry___healthy": {
        "symptoms": ["Vibrant green leaves", "Healthy fruit development", "No disease signs"],
        "causes": ["Proper care and management"],
        "treatments": {
            "chemical": [],
            "organic": [],
            "prevention": ["Regular monitoring", "Proper irrigation", "Balanced fertilization"]
        }
    },
    
    # Cherry
    "Cherry_(including_sour)___Powdery_mildew": {
        "symptoms": ["White powdery coating on leaves", "Leaf curling and distortion", "Reduced fruit quality"],
        "causes": ["Fungal infection by Podosphaera clandestina", "Warm dry days with cool nights", "Poor air circulation"],
        "treatments": {
            "chemical": ["Apply Myclobutanil or Propiconazole", "Sulfur-based fungicides"],
            "organic": ["Neem oil spray", "Baking soda solution", "Milk spray (1:9 ratio)"],
            "prevention": ["Prune for air circulation", "Avoid overhead watering", "Plant resistant varieties"]
        }
    },
    
    "Cherry_(including_sour)___healthy": {
        "symptoms": ["Healthy foliage", "Normal fruit production", "No disease symptoms"],
        "causes": ["Good management practices"],
        "treatments": {
            "chemical": [],
            "organic": [],
            "prevention": ["Continue current practices", "Monitor regularly", "Maintain tree vigor"]
        }
    },
    
    # Corn (Maize)
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "symptoms": ["Small rectangular gray lesions", "Lesions parallel to leaf veins", "Premature leaf death"],
        "causes": ["Fungal infection by Cercospora zeae-maydis", "Warm humid weather", "Continuous corn cropping"],
        "treatments": {
            "chemical": ["Apply Azoxystrobin or Pyraclostrobin", "Triazole fungicides"],
            "organic": ["Remove infected leaves", "Improve drainage", "Crop rotation"],
            "prevention": ["Plant resistant hybrids", "Rotate crops", "Bury crop residue"]
        }
    },
    
    "Corn_(maize)___Common_rust_": {
        "symptoms": ["Small circular reddish-brown pustules", "Pustules on both leaf surfaces", "Premature leaf death"],
        "causes": ["Fungal infection by Puccinia sorghi", "Moderate temperatures", "High humidity"],
        "treatments": {
            "chemical": ["Apply Propiconazole or Azoxystrobin", "Triazole fungicides"],
            "organic": ["Neem oil spray", "Sulfur dust", "Remove heavily infected leaves"],
            "prevention": ["Plant resistant hybrids", "Timely planting", "Balanced fertilization"]
        }
    },
    
    "Corn_(maize)___Northern_Leaf_Blight": {
        "symptoms": ["Long elliptical gray-green lesions", "Lesions turn tan with dark borders", "Severe leaf blighting"],
        "causes": ["Fungal infection by Exserohilum turcicum", "Moderate temperatures", "High humidity"],
        "treatments": {
            "chemical": ["Apply Propiconazole", "Azoxystrobin + Propiconazole combination"],
            "organic": ["Neem oil", "Copper fungicides", "Remove infected parts"],
            "prevention": ["Plant resistant hybrids", "Crop rotation", "Deep plow residue"]
        }
    },
    
    "Corn_(maize)___healthy": {
        "symptoms": ["Healthy green leaves", "No lesions", "Normal growth"],
        "causes": ["Good agronomic practices"],
        "treatments": {
            "chemical": [],
            "organic": [],
            "prevention": ["Continue monitoring", "Maintain nutrition", "Proper water management"]
        }
    },
    
    # Grape
    "Grape___Black_rot": {
        "symptoms": ["Circular tan lesions on leaves", "Black mummified berries", "Fruit rot"],
        "causes": ["Fungal infection by Guignardia bidwellii", "Warm wet weather", "Poor sanitation"],
        "treatments": {
            "chemical": ["Apply Mancozeb or Captan", "Myclobutanil fungicide"],
            "organic": ["Copper fungicide", "Remove mummified berries", "Improve air circulation"],
            "prevention": ["Remove infected fruit", "Prune for airflow", "Fungicide sprays"]
        }
    },
    
    "Grape___Esca_(Black_Measles)": {
        "symptoms": ["Tiger-stripe pattern on leaves", "Dark streaks in wood", "Berry spotting"],
        "causes": ["Complex fungal infection", "Pruning wounds", "Old vines"],
        "treatments": {
            "chemical": ["No effective chemical control", "Prune infected wood"],
            "organic": ["Remove infected vines", "Improve vine nutrition", "Reduce stress"],
            "prevention": ["Protect pruning wounds", "Maintain vine health", "Avoid water stress"]
        }
    },
    
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "symptoms": ["Angular brown spots on leaves", "Premature defoliation", "Reduced yield"],
        "causes": ["Fungal infection", "Warm humid conditions", "Poor air circulation"],
        "treatments": {
            "chemical": ["Apply Mancozeb or Copper fungicides"],
            "organic": ["Neem oil spray", "Remove infected leaves", "Improve drainage"],
            "prevention": ["Prune for airflow", "Avoid overhead irrigation", "Fungicide applications"]
        }
    },
    
    "Grape___healthy": {
        "symptoms": ["Healthy green foliage", "Normal fruit development", "No disease signs"],
        "causes": ["Good vineyard management"],
        "treatments": {
            "chemical": [],
            "organic": [],
            "prevention": ["Continue monitoring", "Maintain vine health", "Good sanitation"]
        }
    },
    
    # Orange
    "Orange___Haunglongbing_(Citrus_greening)": {
        "symptoms": ["Yellow shoots", "Blotchy mottled leaves", "Lopsided bitter fruits", "Tree decline"],
        "causes": ["Bacterial infection spread by Asian citrus psyllid", "No cure available"],
        "treatments": {
            "chemical": ["Control psyllid vectors with insecticides", "Nutrient management"],
            "organic": ["Remove infected trees", "Control psyllid population", "Plant disease-free trees"],
            "prevention": ["Use certified disease-free nursery stock", "Control psyllid vectors", "Remove infected trees immediately"]
        }
    },
    
    # Peach
    "Peach___Bacterial_spot": {
        "symptoms": ["Small dark spots on leaves", "Sunken lesions on fruits", "Premature leaf drop"],
        "causes": ["Bacterial infection by Xanthomonas", "Warm wet weather", "Wind-driven rain"],
        "treatments": {
            "chemical": ["Apply copper-based bactericides", "Oxytetracycline sprays"],
            "organic": ["Copper sulfate", "Remove infected parts", "Improve air circulation"],
            "prevention": ["Plant resistant varieties", "Avoid overhead irrigation", "Prune for airflow"]
        }
    },
    
    "Peach___healthy": {
        "symptoms": ["Healthy foliage", "Normal fruit development", "No disease symptoms"],
        "causes": ["Good orchard management"],
        "treatments": {
            "chemical": [],
            "organic": [],
            "prevention": ["Continue monitoring", "Maintain tree health", "Good sanitation"]
        }
    },
    
    # Pepper (Bell)
    "Pepper,_bell___Bacterial_spot": {
        "symptoms": ["Small dark spots with yellow halos", "Leaf drop", "Fruit lesions"],
        "causes": ["Bacterial infection by Xanthomonas", "Warm humid weather", "Water splash"],
        "treatments": {
            "chemical": ["Apply copper-based bactericides", "Mancozeb + Copper"],
            "organic": ["Copper sulfate spray", "Remove infected plants", "Improve drainage"],
            "prevention": ["Use disease-free seeds", "Crop rotation", "Avoid overhead watering"]
        }
    },
    
    "Pepper,_bell___healthy": {
        "symptoms": ["Healthy green foliage", "Normal fruit production", "No disease signs"],
        "causes": ["Good cultural practices"],
        "treatments": {
            "chemical": [],
            "organic": [],
            "prevention": ["Continue monitoring", "Maintain plant health", "Good sanitation"]
        }
    },
    
    # Potato
    "Potato___Early_blight": {
        "symptoms": ["Dark brown spots with concentric rings", "Yellowing around spots", "Stem lesions", "Fruit rot"],
        "causes": ["Fungal infection by Alternaria solani", "Warm humid conditions", "Poor air circulation"],
        "treatments": {
            "chemical": ["Apply Mancozeb or Chlorothalonil", "Azoxystrobin fungicide"],
            "organic": ["Neem oil spray", "Baking soda solution", "Remove infected leaves"],
            "prevention": ["Use disease-free seed potatoes", "Crop rotation", "Adequate spacing"]
        }
    },
    
    "Potato___Late_blight": {
        "symptoms": ["Water-soaked dark lesions", "White mold on undersides", "Rapid spread", "Tuber rot"],
        "causes": ["Oomycete Phytophthora infestans", "Cool moist weather", "High humidity"],
        "treatments": {
            "chemical": ["Apply Metalaxyl or Cymoxanil immediately", "Chlorothalonil + Metalaxyl"],
            "organic": ["Copper fungicide (Bordeaux mixture)", "Remove infected plants", "Improve drainage"],
            "prevention": ["Plant resistant varieties", "Avoid overhead irrigation", "Hill up soil around plants"]
        }
    },
    
    "Potato___healthy": {
        "symptoms": ["Vibrant green foliage", "No disease symptoms", "Healthy growth"],
        "causes": ["Good cultural practices"],
        "treatments": {
            "chemical": [],
            "organic": [],
            "prevention": ["Continue monitoring", "Maintain soil health", "Good field sanitation"]
        }
    },
    
    # Raspberry
    "Raspberry___healthy": {
        "symptoms": ["Healthy canes and foliage", "Normal fruit production", "No disease signs"],
        "causes": ["Good management practices"],
        "treatments": {
            "chemical": [],
            "organic": [],
            "prevention": ["Continue monitoring", "Maintain plant health", "Good sanitation"]
        }
    },
    
    # Soybean
    "Soybean___healthy": {
        "symptoms": ["Healthy green leaves", "Normal pod development", "No disease symptoms"],
        "causes": ["Good agronomic practices"],
        "treatments": {
            "chemical": [],
            "organic": [],
            "prevention": ["Continue monitoring", "Maintain nutrition", "Crop rotation"]
        }
    },
    
    # Squash
    "Squash___Powdery_mildew": {
        "symptoms": ["White powdery coating on leaves", "Leaf yellowing", "Reduced fruit quality"],
        "causes": ["Fungal infection", "Warm dry days with cool nights", "Poor air circulation"],
        "treatments": {
            "chemical": ["Apply Myclobutanil or Sulfur fungicides"],
            "organic": ["Neem oil spray", "Baking soda solution", "Milk spray"],
            "prevention": ["Plant resistant varieties", "Improve air circulation", "Avoid overhead watering"]
        }
    },
    
    # Strawberry
    "Strawberry___Leaf_scorch": {
        "symptoms": ["Purple to brown spots on leaves", "Scorched appearance", "Reduced yield"],
        "causes": ["Fungal infection by Diplocarpon earlianum", "Warm wet weather", "Poor air circulation"],
        "treatments": {
            "chemical": ["Apply Captan or Myclobutanil fungicides"],
            "organic": ["Remove infected leaves", "Improve air circulation", "Neem oil spray"],
            "prevention": ["Plant resistant varieties", "Renovate beds after harvest", "Fungicide applications"]
        }
    },
    
    "Strawberry___healthy": {
        "symptoms": ["Healthy green foliage", "Normal fruit production", "No disease signs"],
        "causes": ["Good management practices"],
        "treatments": {
            "chemical": [],
            "organic": [],
            "prevention": ["Continue monitoring", "Maintain plant health", "Good sanitation"]
        }
    },
    
    # Tomato
    "Tomato___Bacterial_spot": {
        "symptoms": ["Small dark spots with yellow halos", "Raised brown spots on fruits", "Leaf drop"],
        "causes": ["Bacterial infection by Xanthomonas", "Warm humid weather", "Water splash"],
        "treatments": {
            "chemical": ["Apply copper-based bactericides", "Mancozeb + Copper combination"],
            "organic": ["Copper sulfate spray", "Remove infected parts", "Improve air circulation"],
            "prevention": ["Use certified disease-free seeds", "Crop rotation", "Avoid overhead irrigation"]
        }
    },
    
    "Tomato___Early_blight": {
        "symptoms": ["Dark brown spots with concentric rings", "Yellowing around spots", "Stem lesions", "Fruit rot"],
        "causes": ["Fungal infection by Alternaria solani", "Warm humid conditions", "Poor air circulation"],
        "treatments": {
            "chemical": ["Apply Chlorothalonil or Mancozeb", "Azoxystrobin fungicide"],
            "organic": ["Neem oil spray", "Baking soda solution", "Copper fungicide"],
            "prevention": ["Use disease-free transplants", "Mulch to prevent soil splash", "Stake plants"]
        }
    },
    
    "Tomato___Late_blight": {
        "symptoms": ["Large brown water-soaked lesions", "White mold on undersides", "Brown streaks on stems", "Firm brown rot on fruits"],
        "causes": ["Oomycete Phytophthora infestans", "Cool moist weather", "High humidity"],
        "treatments": {
            "chemical": ["Apply Metalaxyl or Cymoxanil immediately", "Chlorothalonil + Metalaxyl"],
            "organic": ["Copper fungicide (Bordeaux mixture)", "Remove infected plants", "Improve air circulation"],
            "prevention": ["Plant resistant varieties", "Avoid overhead irrigation", "Provide adequate spacing"]
        }
    },
    
    "Tomato___Leaf_Mold": {
        "symptoms": ["Pale yellow spots on upper leaf surface", "Olive-green to gray mold on undersides", "Leaf curling"],
        "causes": ["Fungal infection by Passalora fulva", "High humidity", "Poor air circulation"],
        "treatments": {
            "chemical": ["Apply Chlorothalonil or Mancozeb fungicides"],
            "organic": ["Improve ventilation", "Reduce humidity", "Remove infected leaves"],
            "prevention": ["Plant resistant varieties", "Improve air circulation", "Avoid overhead watering"]
        }
    },
    
    "Tomato___Septoria_leaf_spot": {
        "symptoms": ["Small circular spots with gray centers", "Dark borders around spots", "Premature defoliation"],
        "causes": ["Fungal infection by Septoria lycopersici", "Warm wet weather", "Splash from rain or irrigation"],
        "treatments": {
            "chemical": ["Apply Chlorothalonil or Mancozeb", "Copper fungicides"],
            "organic": ["Remove infected leaves", "Mulch to prevent splash", "Neem oil spray"],
            "prevention": ["Crop rotation", "Avoid overhead watering", "Stake plants for airflow"]
        }
    },
    
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "symptoms": ["Yellow stippling on leaves", "Fine webbing", "Leaf bronzing", "Premature leaf drop"],
        "causes": ["Spider mite infestation", "Hot dry weather", "Dusty conditions"],
        "treatments": {
            "chemical": ["Apply Abamectin or Spiromesifen miticides"],
            "organic": ["Neem oil spray", "Insecticidal soap", "Strong water spray", "Predatory mites"],
            "prevention": ["Maintain adequate moisture", "Avoid water stress", "Encourage beneficial insects"]
        }
    },
    
    "Tomato___Target_Spot": {
        "symptoms": ["Brown spots with concentric rings", "Target-like appearance", "Premature defoliation"],
        "causes": ["Fungal infection by Corynespora cassiicola", "Warm humid weather", "Poor air circulation"],
        "treatments": {
            "chemical": ["Apply Chlorothalonil or Mancozeb fungicides"],
            "organic": ["Remove infected leaves", "Improve air circulation", "Copper fungicides"],
            "prevention": ["Plant resistant varieties", "Crop rotation", "Avoid overhead watering"]
        }
    },
    
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "symptoms": ["Upward leaf curling", "Yellowing of leaves", "Stunted growth", "Reduced fruit set"],
        "causes": ["Viral infection spread by whiteflies", "No cure available"],
        "treatments": {
            "chemical": ["Control whitefly vectors with insecticides", "Remove infected plants"],
            "organic": ["Yellow sticky traps", "Reflective mulches", "Remove infected plants immediately"],
            "prevention": ["Use virus-free transplants", "Control whitefly population", "Use insect-proof nets"]
        }
    },
    
    "Tomato___Tomato_mosaic_virus": {
        "symptoms": ["Mottled light and dark green on leaves", "Leaf distortion", "Stunted growth", "Reduced yield"],
        "causes": ["Viral infection spread mechanically", "Contaminated tools", "Infected seeds"],
        "treatments": {
            "chemical": ["No chemical control available", "Remove infected plants"],
            "organic": ["Remove infected plants immediately", "Disinfect tools", "Wash hands"],
            "prevention": ["Use virus-free seeds", "Disinfect tools between plants", "Avoid tobacco use near plants"]
        }
    },
    
    "Tomato___healthy": {
        "symptoms": ["Healthy dark green foliage", "Normal fruit development", "No disease symptoms"],
        "causes": ["Good cultural practices"],
        "treatments": {
            "chemical": [],
            "organic": [],
            "prevention": ["Continue monitoring", "Maintain plant health", "Good sanitation"]
        }
    }
}

def get_disease_info(disease_class: str) -> dict:
    """
    Get disease-specific information for a given disease class
    Returns crop-specific generic info if disease not in database
    """
    # Normalize the class name
    normalized = disease_class.replace(' - ', '___').replace(' ', '_')
    
    # Try to find exact match
    if normalized in DISEASE_DATABASE:
        return DISEASE_DATABASE[normalized]
    
    # Try to find partial match
    for key in DISEASE_DATABASE.keys():
        if key.lower() in normalized.lower() or normalized.lower() in key.lower():
            return DISEASE_DATABASE[key]
    
    # Extract crop type from disease_class
    crop_type = disease_class.split(' - ')[0] if ' - ' in disease_class else disease_class.split('___')[0]
    
    # Return crop-specific generic info based on detected issue
    if "fungal" in disease_class.lower() or "rot" in disease_class.lower():
        return {
            "symptoms": [
                f"Dark or brown lesions on {crop_type.lower()} leaves or fruits",
                "Water-soaked spots that expand over time",
                "Possible mold or fungal growth",
                "Premature leaf drop or fruit decay"
            ],
            "causes": [
                "Fungal infection (specific pathogen unknown)",
                "High humidity and moisture",
                "Poor air circulation",
                "Wounds or injuries on plant"
            ],
            "treatments": {
                "chemical": [
                    f"Apply broad-spectrum fungicide (Mancozeb or Chlorothalonil) for {crop_type.lower()}",
                    "Copper-based fungicides as preventive measure",
                    "Follow label instructions for application timing"
                ],
                "organic": [
                    "Neem oil spray (2-5% solution) applied weekly",
                    "Baking soda solution (1 tablespoon per gallon water)",
                    "Remove and destroy infected plant parts immediately",
                    "Improve air circulation around plants"
                ],
                "prevention": [
                    f"Use disease-resistant {crop_type.lower()} varieties",
                    "Practice crop rotation (3-4 year cycle)",
                    "Avoid overhead watering - water at base of plants",
                    "Maintain proper plant spacing for air circulation",
                    "Remove plant debris and fallen leaves regularly"
                ]
            }
        }
    elif "chlorosis" in disease_class.lower() or "nutrient" in disease_class.lower():
        return {
            "symptoms": [
                f"Yellowing of {crop_type.lower()} leaves (chlorosis)",
                "Stunted growth",
                "Pale or light green foliage",
                "Reduced vigor and yield"
            ],
            "causes": [
                "Nutrient deficiency (nitrogen, iron, or magnesium)",
                "Poor soil pH affecting nutrient availability",
                "Waterlogged or compacted soil",
                "Root damage or disease"
            ],
            "treatments": {
                "chemical": [
                    f"Apply balanced NPK fertilizer suitable for {crop_type.lower()}",
                    "Foliar spray with micronutrients (iron, zinc, manganese)",
                    "Soil pH adjustment if needed (lime or sulfur)"
                ],
                "organic": [
                    "Compost or well-rotted manure application",
                    "Fish emulsion or seaweed extract foliar spray",
                    "Bone meal for phosphorus, blood meal for nitrogen",
                    "Epsom salt (magnesium sulfate) for magnesium deficiency"
                ],
                "prevention": [
                    "Regular soil testing to monitor nutrient levels",
                    "Maintain proper soil pH for optimal nutrient uptake",
                    "Ensure good drainage to prevent waterlogging",
                    "Apply organic mulch to improve soil health"
                ]
            }
        }
    elif "healthy" in disease_class.lower():
        return {
            "symptoms": [
                f"Healthy {crop_type.lower()} with vibrant foliage",
                "No visible disease symptoms",
                "Normal growth and development"
            ],
            "causes": [
                "Good cultural practices",
                "Proper disease management"
            ],
            "treatments": {
                "chemical": [],
                "organic": [],
                "prevention": [
                    "Continue regular monitoring for early disease detection",
                    "Maintain current good practices",
                    "Practice preventive disease management"
                ]
            }
        }
    else:
        # Generic fallback
        return {
            "symptoms": [
                f"Abnormal appearance on {crop_type.lower()} plant",
                "Possible disease or stress symptoms",
                "Consult agricultural expert for accurate diagnosis"
            ],
            "causes": [
                "Disease or environmental stress (specific cause unknown)",
                "May require professional diagnosis"
            ],
            "treatments": {
                "chemical": [
                    "Consult local agricultural extension for specific recommendations",
                    f"Use appropriate fungicide or pesticide for {crop_type.lower()}"
                ],
                "organic": [
                    "Remove affected plant parts",
                    "Improve plant growing conditions",
                    "Maintain good plant hygiene"
                ],
                "prevention": [
                    "Monitor plants regularly for early detection",
                    "Practice good sanitation and crop rotation",
                    "Maintain optimal growing conditions"
                ]
            }
        }
