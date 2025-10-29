import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useRef, useState, useEffect } from "react";
import { useToast } from "@/hooks/use-toast";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Bug, AlertTriangle, Shield, Search, Camera, Leaf, CheckCircle, Upload } from "lucide-react";

interface DiseaseResult {
  disease_name: string;
  crop_type: string;
  confidence: number;
  severity: string;
  symptoms: string[];
  causes: string[];
  treatments: {
    chemical: string[];
    organic: string[];
    prevention: string[];
  };
  prevention: string[];
  top3_predictions?: Array<{
    class: string;
    confidence: number;
  }>;
  model_info?: {
    model_name: string;
    total_classes: number;
    prediction_time: string;
  };
}

const PestControl = () => {
  const { toast } = useToast();
  const [selectedPlantType, setSelectedPlantType] = useState<string>("");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<DiseaseResult | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(document.createElement('input'));
  useEffect(() => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'image/jpeg,image/png,image/jpg';
    input.style.display = 'none';
    input.onchange = (e: Event) => {
      const target = e.target as HTMLInputElement;
      const file = target.files?.[0];
      if (file) {
        handleFileSelect(file);
      }
    };
    document.body.appendChild(input);
    fileInputRef.current = input;
    
    return () => {
      document.body.removeChild(input);
    };
  }, []);

  const commonPests = [
    {
      name: "Brown Plant Hopper",
      crop: "Rice",
      severity: "High",
      symptoms: "Yellow patches on leaves, stunted growth",
      treatment: "Use BPH resistant varieties, apply neem oil",
      prevention: "Maintain water level, avoid excess nitrogen",
      image: "🦗"
    },
    {
      name: "Aphids", 
      crop: "Wheat",
      severity: "Medium",
      symptoms: "Curled leaves, sticky honeydew, yellowing",
      treatment: "Spray insecticidal soap or neem oil",
      prevention: "Encourage beneficial insects, avoid over-fertilization",
      image: "🐛"
    },
    {
      name: "Bollworm",
      crop: "Cotton",
      severity: "High", 
      symptoms: "Holes in bolls, damaged flowers and buds",
      treatment: "Use pheromone traps, apply Bt spray",
      prevention: "Crop rotation, remove plant debris",
      image: "🐛"
    },
    {
      name: "Stem Borer",
      crop: "Sugarcane",
      severity: "Medium",
      symptoms: "Dead hearts, bore holes in stems",
      treatment: "Apply carbofuran, use resistant varieties",
      prevention: "Destroy stubble, proper field sanitation",
      image: "🪲"
    }
  ];

  const diseases = [
    {
      name: "Blast Disease",
      crop: "Rice",
      type: "Fungal",
      symptoms: "Spindle-shaped lesions on leaves",
      treatment: "Apply Tricyclazole fungicide",
      prevention: "Use resistant varieties, balanced fertilization"
    },
    {
      name: "Rust",
      crop: "Wheat", 
      type: "Fungal",
      symptoms: "Orange pustules on leaves and stems",
      treatment: "Apply Propiconazole spray",
      prevention: "Crop rotation, timely sowing"
    },
    {
      name: "Wilt",
      crop: "Cotton",
      type: "Fungal",
      symptoms: "Yellowing and wilting of plants",
      treatment: "Soil drenching with fungicide",
      prevention: "Use disease-free seeds, soil fumigation"
    }
  ];

  const getSeverityColor = (severity: string) => {
    switch (severity.toLowerCase()) {
      case 'high':
        return 'destructive';
      case 'medium':
        return 'default';
      case 'low':
        return 'secondary';
      default:
        return 'secondary';
    }
  };

  const handleFileSelect = (file: File) => {
    if (!file) return;
    
    // Check file type
    const validTypes = ['image/jpeg', 'image/png', 'image/jpg'];
    if (!validTypes.includes(file.type)) {
      toast({
        title: "Invalid file type",
        description: "Please upload a JPG or PNG image",
        variant: "destructive",
      });
      return;
    }

    // Check file size (5MB limit)
    if (file.size > 5 * 1024 * 1024) {
      toast({
        title: "File too large",
        description: "Maximum file size is 5MB",
        variant: "destructive",
      });
      return;
    }

    setSelectedFile(file);
    setAnalysisResult(null);
    
    // Create image preview
    const reader = new FileReader();
    reader.onloadend = () => {
      setImagePreview(reader.result as string);
    };
    reader.readAsDataURL(file);
  };

  const clearFile = () => {
    setSelectedFile(null);
    setImagePreview(null);
    setAnalysisResult(null);
    setSelectedPlantType(""); // Also clear plant type
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const analyzeImage = async (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    
    if (!selectedFile) {
      toast({
        title: "No image selected",
        description: "Please select an image first",
        variant: "destructive",
      });
      return;
    }

    if (!selectedPlantType) {
      toast({
        title: "No plant type selected",
        description: "Please select the plant type first",
        variant: "destructive",
      });
      return;
    }

    setIsAnalyzing(true);
    setAnalysisResult(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      // Call our FastAPI backend
      const response = await fetch(`http://localhost:8000/predict?plant_type=${encodeURIComponent(selectedPlantType)}`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      // Handle different response types
      if (data.class === 'invalid_image') {
        toast({
          title: "Invalid Image",
          description: data.message || "The uploaded image does not appear to be a plant, leaf, or fruit.",
          variant: "destructive",
        });
        clearFile();
        return;
      }
      
      // Format the result to match the expected interface
      const confidence = data.confidence; // Already rounded from backend
      const result: DiseaseResult = {
        disease_name: data.class.replace('___', ' - ').replace(/_/g, ' '),
        crop_type: data.class.split('___')[0].replace(/_/g, ' '),
        confidence: confidence,
        severity: confidence > 80 ? "High" : confidence > 60 ? "Medium" : "Low",
        // Use disease-specific information from backend
        symptoms: data.symptoms || [
          "Yellowing or browning of leaves",
          "Spots or lesions on leaves",
          "Wilting or drooping"
        ],
        causes: data.causes || [
          "Fungal or bacterial infection",
          "Environmental stress",
          "Nutrient deficiencies"
        ],
        treatments: data.treatments || {
          chemical: [
            "Fungicide application (e.g., Chlorothalonil for fungal diseases)",
            "Bactericides for bacterial infections"
          ],
          organic: [
            "Neem oil spray (2-5% solution)",
            "Baking soda spray (1 tsp per quart of water)",
            "Proper plant spacing for air circulation"
          ],
          prevention: [
            "Use certified disease-free seeds",
            "Practice crop rotation (3-4 year cycle)",
            "Water at the base of plants"
          ]
        },
        prevention: data.treatments?.prevention || [
          "Regular field monitoring (check plants weekly)",
          "Use resistant varieties when available",
          "Avoid overhead watering to reduce leaf wetness"
        ],
        model_info: {
          model_name: "EfficientNet-B1",
          total_classes: 39,
          prediction_time: new Date().toISOString()
        },
        // Include top predictions from the backend
        top3_predictions: data.top3_predictions?.map((p: any) => ({
          class: p.class.replace('___', ' - ').replace(/_/g, ' '),
          confidence: p.confidence
        }))
      };

      setAnalysisResult(result);

      // Show toast with message from backend if any
      if (data.message) {
        toast({
          title: "Analysis Complete",
          description: data.message,
          variant: "default",
        });
      } else {
        toast({
          title: "Analysis Complete",
          description: `Detected: ${result.disease_name} with ${result.confidence.toFixed(1)}% confidence`,
        });
      }

    } catch (error) {
      console.error('Error analyzing image:', error);
      toast({
        title: "Analysis Failed",
        description: error instanceof Error ? error.message : "Failed to analyze the image. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="min-h-screen bg-background">
      
      <div className="w-full px-4 py-8">
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-2">Pest & Disease Control</h1>
          <p className="text-muted-foreground text-lg">
            AI-powered plant disease detection and management
          </p>
          <div className="mt-4 p-4 bg-gradient-to-r from-green-50 to-blue-50 border border-green-200 rounded-lg">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
              <span className="text-sm font-medium text-green-800">
                 HIGH-ACCURACY AI SYSTEM: Deep Learning • Transfer Learning • 39 Disease Classes
              </span>
            </div>
            <p className="text-xs text-green-600 mt-1">
                EfficientNet Architecture • Data Augmentation • Real-time Analysis • Backend: localhost:8000
            </p>
          </div>
        </div>

        {/* High-Accuracy Disease Detection */}
        <Card className="mb-8">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Camera className="h-5 w-5" />
              AI Disease Detection
            </CardTitle>
            <CardDescription>
              Upload a photo of your plant for instant disease identification using deep learning
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {/* Plant Selection Section */}
              <div>
                <h4 className="font-semibold mb-4">Select Plant Type</h4>
                <div className="mb-6">
                  <Select value={selectedPlantType} onValueChange={setSelectedPlantType}>
                    <SelectTrigger className="w-full">
                      <SelectValue placeholder="Choose the plant type..." />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="tomato">Tomato</SelectItem>
                      <SelectItem value="potato">Potato</SelectItem>
                      <SelectItem value="corn">Corn (Maize)</SelectItem>
                      <SelectItem value="grape">Grape</SelectItem>
                      <SelectItem value="apple">Apple</SelectItem>
                      <SelectItem value="pepper">Bell Pepper</SelectItem>
                      <SelectItem value="strawberry">Strawberry</SelectItem>
                      <SelectItem value="peach">Peach</SelectItem>
                      <SelectItem value="cherry">Cherry</SelectItem>
                      <SelectItem value="blueberry">Blueberry</SelectItem>
                    </SelectContent>
                  </Select>
                  {selectedPlantType && (
                    <div className="flex items-center justify-between mt-2">
                      <p className="text-sm text-muted-foreground">
                        Selected: <span className="font-medium capitalize">{selectedPlantType}</span>
                      </p>
                      <Button 
                        variant="outline" 
                        size="sm" 
                        onClick={() => setSelectedPlantType("")}
                      >
                        Clear
                      </Button>
                    </div>
                  )}
                </div>

                {/* Image Upload Section */}
                <h4 className="font-semibold mb-4">Upload Plant Image</h4>
                
                <div 
                  className="border-2 border-dashed border-muted-foreground/25 rounded-lg p-6 text-center cursor-pointer hover:bg-accent/10 transition-colors"
                  onClick={() => fileInputRef.current?.click()}
                  style={{ minHeight: '200px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}
                >
                  {selectedFile ? (
                    <div className="relative w-full">
                      <div className="flex flex-col items-center">
                        {imagePreview ? (
                          <div className="relative mb-4">
                            <img 
                              src={imagePreview} 
                              alt="Preview" 
                              className="max-h-48 rounded-md object-contain border"
                            />
                            <Button 
                              variant="destructive" 
                              size="sm" 
                              className="absolute -top-3 -right-3 rounded-full p-1 h-8 w-8"
                              onClick={(e) => {
                                e.stopPropagation();
                                clearFile();
                              }}
                            >
                              ×
                            </Button>
                          </div>
                        ) : (
                          <div className="space-y-2 text-center py-4">
                            <CheckCircle className="h-8 w-8 mx-auto text-green-500" />
                            <p className="font-medium">{selectedFile.name}</p>
                            <p className="text-sm text-muted-foreground">
                              Size: {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                            </p>
                          </div>
                        )}
                        <div className="flex gap-2 w-full">
                          <Button 
                            onClick={analyzeImage} 
                            disabled={isAnalyzing} 
                            className="flex-1"
                          >
                            {isAnalyzing ? (
                              <>
                                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                                Analyzing...
                              </>
                            ) : (
                              <>
                                <Search className="h-4 w-4 mr-2" />
                                Analyze Disease
                              </>
                            )}
                          </Button>
                          <Button variant="outline" onClick={clearFile}>
                            Clear
                          </Button>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="space-y-2 text-center">
                      <Upload className="w-8 h-8 mx-auto text-muted-foreground" />
                      <p className="text-sm text-muted-foreground">
                        Drag and drop an image here, or click to select
                      </p>
                      <p className="text-xs text-muted-foreground">
                        Supported formats: JPG, PNG (max 5MB)
                      </p>
                    </div>
                  )}
                  {/* File input is now managed by useEffect */}
                </div>

                {/* Removed duplicate analyze button */}
              </div>

              {/* Analysis Results Section */}
              <div>
                <h4 className="font-semibold mb-4">Analysis Results</h4>
                
                {!analysisResult ? (
                  <div className="text-center py-12">
                    <Camera className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
                    <p className="text-muted-foreground">
                      Upload a plant image to get started
                    </p>
                    <p className="text-sm text-muted-foreground mt-2">
                      High-accuracy AI disease detection system ready!
                    </p>
                  </div>
                ) : (
                  <div className="space-y-4">
                    <Alert className={analysisResult.disease_name.toLowerCase().includes('healthy') ? "border-green-200 bg-green-50" : "border-red-200 bg-red-50"}>
                      <div className="flex items-center gap-2">
                        {analysisResult.disease_name.toLowerCase().includes('healthy') ? (
                          <CheckCircle className="h-4 w-4 text-green-600" />
                        ) : (
                          <AlertTriangle className="h-4 w-4 text-red-600" />
                        )}
                        <AlertDescription className="font-medium">
                          {analysisResult.disease_name.toLowerCase().includes('healthy') 
                            ? "Plant Status: Healthy" 
                            : `Disease Detected: ${analysisResult.disease_name}`
                          }
                        </AlertDescription>
                      </div>
                    </Alert>

                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <p className="text-sm font-medium">Confidence</p>
                        <p className="text-2xl font-bold text-primary">
                          {analysisResult.confidence.toFixed(1)}%
                        </p>
                      </div>
                      <div>
                        <p className="text-sm font-medium">Status</p>
                        <Badge variant={analysisResult.disease_name.toLowerCase().includes('healthy') ? "default" : getSeverityColor(analysisResult.severity)}>
                          {analysisResult.disease_name.toLowerCase().includes('healthy') ? "Healthy" : analysisResult.severity}
                        </Badge>
                      </div>
                    </div>

                    <div>
                      <p className="text-sm font-medium mb-2">Crop Type</p>
                      <Badge variant="outline" className="capitalize">
                        {analysisResult.crop_type}
                      </Badge>
                    </div>

                    <div>
                      <p className="text-sm font-medium mb-2">Symptoms</p>
                      <ul className="text-sm space-y-1">
                        {analysisResult.symptoms.map((symptom, idx) => (
                          <li key={idx} className="flex items-center gap-2">
                            <div className="w-1 h-1 bg-red-500 rounded-full" />
                            {symptom}
                          </li>
                        ))}
                      </ul>
                    </div>

                    <div>
                      <p className="text-sm font-medium mb-2">Causes</p>
                      <ul className="text-sm space-y-1">
                        {analysisResult.causes.map((cause, idx) => (
                          <li key={idx} className="flex items-center gap-2">
                            <div className="w-1 h-1 bg-orange-500 rounded-full" />
                            {cause}
                          </li>
                        ))}
                      </ul>
                    </div>

                    {/* Treatment Recommendations */}
                    <div className="space-y-3">
                      <h5 className="font-medium">Treatment Recommendations</h5>
                      
                      <div>
                        <p className="text-sm font-medium text-blue-600 mb-1">Chemical Treatments</p>
                        <ul className="text-sm space-y-1">
                          {analysisResult.treatments.chemical.map((treatment, idx) => (
                            <li key={idx} className="flex items-center gap-2">
                              <div className="w-1 h-1 bg-blue-500 rounded-full" />
                              {treatment}
                            </li>
                          ))}
                        </ul>
                      </div>

                      <div>
                        <p className="text-sm font-medium text-green-600 mb-1">Organic Treatments</p>
                        <ul className="text-sm space-y-1">
                          {analysisResult.treatments.organic.map((treatment, idx) => (
                            <li key={idx} className="flex items-center gap-2">
                              <div className="w-1 h-1 bg-green-500 rounded-full" />
                              {treatment}
                            </li>
                          ))}
                        </ul>
                      </div>

                      <div>
                        <p className="text-sm font-medium text-purple-600 mb-1">Prevention Measures</p>
                        <ul className="text-sm space-y-1">
                          {analysisResult.treatments.prevention.map((measure, idx) => (
                            <li key={idx} className="flex items-center gap-2">
                              <div className="w-1 h-1 bg-purple-500 rounded-full" />
                              {measure}
                            </li>
                          ))}
                        </ul>
                      </div>
                    </div>

                    {/* Model Info */}
                    {analysisResult.model_info && (
                      <div className="mt-4 p-3 bg-gray-50 rounded-lg">
                        <p className="text-xs text-gray-600">
                          Model: {analysisResult.model_info.model_name} • 
                          Classes: {analysisResult.model_info.total_classes} • 
                          Time: {new Date(analysisResult.model_info.prediction_time).toLocaleTimeString()}
                        </p>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Common Pests */}
        <Card className="mb-8">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Bug className="h-5 w-5" />
              Common Pests
            </CardTitle>
            <CardDescription>Frequently encountered pests and their management</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-4">
              {commonPests.map((pest, index) => (
                <div key={index} className="border rounded-lg p-4">
                  {analysisResult && (
                  <div className="space-y-6">
                    {/* Main Prediction */}
                    <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg border border-green-200 dark:border-green-800">
                      <div className="flex items-start justify-between">
                        <div>
                          <h3 className="font-semibold text-lg">{analysisResult.disease_name}</h3>
                          <p className="text-sm text-muted-foreground">
                            {analysisResult.crop_type} • {analysisResult.severity} confidence
                          </p>
                        </div>
                        <div className="flex items-center gap-2">
                          <div className="w-16 h-2 bg-green-200 dark:bg-green-900 rounded-full overflow-hidden">
                            <div 
                              className="h-full bg-green-500"
                              style={{ width: `${analysisResult.confidence}%` }}
                            />
                          </div>
                          <Badge variant="secondary" className="text-sm font-mono">
                            {analysisResult.confidence.toFixed(1)}%
                          </Badge>
                        </div>
                      </div>
                      
                      {/* Additional Predictions */}
                      {analysisResult.top3_predictions?.length > 1 && (
                        <div className="mt-4 pt-4 border-t border-green-100 dark:border-green-800">
                          <p className="text-sm text-muted-foreground mb-2">Other possibilities:</p>
                          <div className="space-y-2">
                            {analysisResult.top3_predictions.slice(1).map((pred, idx) => (
                              <div key={idx} className="flex items-center justify-between text-sm">
                                <span className="text-muted-foreground">{pred.class}</span>
                                <span className="font-mono">{pred.confidence.toFixed(1)}%</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                  )}
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-4">
                    <div>
                      <h4 className="font-medium text-sm mb-1 flex items-center gap-1">
                        <AlertTriangle className="h-3 w-3" />
                        Symptoms
                      </h4>
                      <p className="text-sm text-muted-foreground">{pest.symptoms}</p>
                    </div>
                    <div>
                      <h4 className="font-medium text-sm mb-1 flex items-center gap-1">
                        <Leaf className="h-3 w-3" />
                        Treatment
                      </h4>
                      <p className="text-sm text-muted-foreground">{pest.treatment}</p>
                    </div>
                    <div>
                      <h4 className="font-medium text-sm mb-1 flex items-center gap-1">
                        <Shield className="h-3 w-3" />
                        Prevention
                      </h4>
                      <p className="text-sm text-muted-foreground">{pest.prevention}</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Diseases */}
        <Card className="mb-8">
          <CardHeader>
            <CardTitle>Common Plant Diseases</CardTitle>
            <CardDescription>Fungal, bacterial, and viral diseases affecting crops</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-4">
              {diseases.map((disease, index) => (
                <div key={index} className="border rounded-lg p-4">
                  <div className="flex items-start justify-between mb-3">
                    <div>
                      <h3 className="font-semibold text-lg">{disease.name}</h3>
                      <div className="flex items-center gap-2 mt-1">
                        <Badge variant="outline">{disease.type}</Badge>
                        <span className="text-sm text-muted-foreground">• {disease.crop}</span>
                      </div>
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div>
                      <h4 className="font-medium text-sm mb-1">Symptoms</h4>
                      <p className="text-sm text-muted-foreground">{disease.symptoms}</p>
                    </div>
                    <div>
                      <h4 className="font-medium text-sm mb-1">Treatment</h4>
                      <p className="text-sm text-muted-foreground">{disease.treatment}</p>
                    </div>
                    <div>
                      <h4 className="font-medium text-sm mb-1">Prevention</h4>
                      <p className="text-sm text-muted-foreground">{disease.prevention}</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Integrated Pest Management */}
        <Card>
          <CardHeader>
            <CardTitle>Integrated Pest Management (IPM)</CardTitle>
            <CardDescription>Sustainable approach to pest control</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="p-4 bg-green-50 rounded-lg">
                <h4 className="font-semibold text-green-800 mb-2">Biological Control</h4>
                <p className="text-sm text-green-700">Use beneficial insects, parasites, and natural predators</p>
              </div>
              <div className="p-4 bg-blue-50 rounded-lg">
                <h4 className="font-semibold text-blue-800 mb-2">Cultural Control</h4>
                <p className="text-sm text-blue-700">Crop rotation, field sanitation, resistant varieties</p>
              </div>
              <div className="p-4 bg-yellow-50 rounded-lg">
                <h4 className="font-semibold text-yellow-800 mb-2">Mechanical Control</h4>
                <p className="text-sm text-yellow-700">Physical removal, traps, barriers</p>
              </div>
              <div className="p-4 bg-purple-50 rounded-lg">
                <h4 className="font-semibold text-purple-800 mb-2">Chemical Control</h4>
                <p className="text-sm text-purple-700">Selective pesticides as last resort</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default PestControl;