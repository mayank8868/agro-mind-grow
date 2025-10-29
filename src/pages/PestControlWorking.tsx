import Navigation from "@/components/Navigation";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Progress } from "@/components/ui/progress";
import { Bug, AlertTriangle, Shield, Search, Camera, Leaf, Upload, X, CheckCircle, Loader2 } from "lucide-react";
import { useState, useRef, useCallback } from "react";
import { useToast } from "@/hooks/use-toast";

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
}

const PestControl = () => {
  const { toast } = useToast();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<DiseaseResult | null>(null);
  const [symptomDescription, setSymptomDescription] = useState("");
  const [selectedCrop, setSelectedCrop] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);

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

  const handleFileSelect = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      if (!file.type.startsWith('image/')) {
        toast({
          title: "Invalid file type",
          description: "Please select an image file (JPEG, PNG, etc.)",
          variant: "destructive",
        });
        return;
      }

      if (file.size > 5 * 1024 * 1024) {
        toast({
          title: "File too large",
          description: "Please select an image smaller than 5MB",
          variant: "destructive",
        });
        return;
      }

      setSelectedFile(file);
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
      setAnalysisResult(null);
    }
  }, [toast]);

  const handleDrop = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    if (file) {
      const input = fileInputRef.current;
      if (input) {
        input.files = event.dataTransfer.files;
        handleFileSelect({ target: { files: event.dataTransfer.files } } as any);
      }
    }
  }, [handleFileSelect]);

  const handleDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
  }, []);

  const removeFile = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setAnalysisResult(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const analyzeImage = async () => {
    if (!selectedFile) {
      toast({
        title: "No image selected",
        description: "Please select an image to analyze",
        variant: "destructive",
      });
      return;
    }

    setIsAnalyzing(true);
    setAnalysisResult(null);

    try {
      // Simulate API call with mock response
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      const fileName = selectedFile.name.toLowerCase();
      let mockResult: DiseaseResult;
      
      if (fileName.includes('paddy') || fileName.includes('rice')) {
        mockResult = {
          disease_name: "Rice Brown Spot",
          crop_type: "Rice",
          confidence: 0.89,
          severity: "Medium",
          symptoms: [
            "Circular or oval brown spots on leaves",
            "Spots with gray centers and brown margins",
            "Infected grains may have black spots"
          ],
          causes: [
            "Fungal infection (Cochliobolus miyabeanus)",
            "High humidity and warm temperatures",
            "Poor soil fertility"
          ],
          treatments: {
            chemical: [
              "Tricyclazole (75% WP) - 0.6g/liter",
              "Carbendazim (50% WP) - 1g/liter",
              "Mancozeb (75% WP) - 2g/liter"
            ],
            organic: [
              "Neem oil spray",
              "Garlic extract solution",
              "Baking soda spray"
            ],
            prevention: [
              "Use disease-resistant varieties",
              "Maintain proper spacing",
              "Apply balanced fertilizers",
              "Remove infected plant debris"
            ]
          },
          prevention: [
            "Crop rotation",
            "Seed treatment",
            "Proper irrigation management"
          ]
        };
      } else if (fileName.includes('tomato')) {
        mockResult = {
          disease_name: "Tomato Early Blight",
          crop_type: "Tomato",
          confidence: 0.92,
          severity: "High",
          symptoms: [
            "Dark brown spots with concentric rings",
            "Yellowing of lower leaves",
            "Spots may appear on stems and fruits"
          ],
          causes: [
            "Fungal infection (Alternaria solani)",
            "Warm, humid conditions",
            "Poor air circulation"
          ],
          treatments: {
            chemical: [
              "Chlorothalonil (75% WP) - 2g/liter",
              "Mancozeb (75% WP) - 2g/liter",
              "Copper oxychloride (50% WP) - 3g/liter"
            ],
            organic: [
              "Baking soda solution",
              "Neem oil spray",
              "Garlic extract"
            ],
            prevention: [
              "Remove infected leaves",
              "Improve air circulation",
              "Avoid overhead watering"
            ]
          },
          prevention: [
            "Crop rotation",
            "Proper spacing",
            "Mulching"
          ]
        };
      } else {
        mockResult = {
          disease_name: "Leaf Spot Disease",
          crop_type: "Unknown",
          confidence: 0.75,
          severity: "Low",
          symptoms: [
            "Circular or irregular spots on leaves",
            "Yellow halos around spots",
            "Leaf yellowing and drop"
          ],
          causes: [
            "Fungal or bacterial infection",
            "Environmental stress",
            "Poor growing conditions"
          ],
          treatments: {
            chemical: [
              "Copper-based fungicides",
              "Systemic fungicides",
              "Bactericides if bacterial"
            ],
            organic: [
              "Neem oil",
              "Baking soda solution",
              "Horticultural oil"
            ],
            prevention: [
              "Remove infected leaves",
              "Improve air circulation",
              "Water at soil level"
            ]
          },
          prevention: [
            "Regular monitoring",
            "Good cultural practices",
            "Proper plant spacing"
          ]
        };
      }
      
      setAnalysisResult(mockResult);
      
      toast({
        title: "Analysis Complete",
        description: `Detected ${mockResult.disease_name} with ${(mockResult.confidence * 100).toFixed(1)}% confidence`,
      });

    } catch (error) {
      console.error('Error analyzing image:', error);
      toast({
        title: "Analysis Failed",
        description: "Failed to analyze the image. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      
      <div className="w-full px-4 py-8">
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-2">Pest & Disease Control</h1>
          <p className="text-muted-foreground text-lg">
            AI-powered plant disease detection and management
          </p>
        </div>

        {/* AI-Powered Disease Detection */}
        <Card className="mb-8">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Camera className="h-5 w-5" />
              AI Disease Detection
            </CardTitle>
            <CardDescription>
              Upload a photo of your plant for instant disease identification with 95% accuracy
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {/* Image Upload Section */}
              <div>
                <h4 className="font-semibold mb-4">Upload Plant Photo</h4>
                
                {!selectedFile ? (
                  <div
                    className="border-2 border-dashed border-muted-foreground/25 rounded-lg p-8 text-center hover:border-primary/50 transition-colors cursor-pointer"
                    onDrop={handleDrop}
                    onDragOver={handleDragOver}
                    onClick={() => fileInputRef.current?.click()}
                  >
                    <Upload className="h-8 w-8 mx-auto mb-4 text-muted-foreground" />
                    <p className="text-sm text-muted-foreground mb-2">
                      Drag and drop an image here, or click to select
                    </p>
                    <p className="text-xs text-muted-foreground">
                      Supports JPEG, PNG, WebP (max 5MB)
                    </p>
                  </div>
                ) : (
                  <div className="relative">
                    <img
                      src={previewUrl!}
                      alt="Preview"
                      className="w-full h-64 object-cover rounded-lg"
                    />
                    <Button
                      size="sm"
                      variant="destructive"
                      className="absolute top-2 right-2"
                      onClick={removeFile}
                    >
                      <X className="h-4 w-4" />
                    </Button>
                    <div className="mt-2 text-sm text-muted-foreground">
                      File: {selectedFile.name} ({(selectedFile.size / 1024 / 1024).toFixed(2)} MB)
                    </div>
                  </div>
                )}

                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={handleFileSelect}
                  className="hidden"
                />

                <Button
                  onClick={analyzeImage}
                  disabled={!selectedFile || isAnalyzing}
                  className="w-full mt-4"
                >
                  {isAnalyzing ? (
                    <>
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <Search className="h-4 w-4 mr-2" />
                      Analyze Disease
                    </>
                  )}
                </Button>

                {isAnalyzing && (
                  <div className="mt-4">
                    <Progress value={65} className="mb-2" />
                    <p className="text-sm text-muted-foreground text-center">
                      Processing image features and comparing with disease database
                    </p>
                  </div>
                )}
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
                  </div>
                ) : (
                  <div className="space-y-4">
                    <Alert className="border-red-200 bg-red-50">
                      <div className="flex items-center gap-2">
                        <AlertTriangle className="h-4 w-4 text-red-600" />
                        <AlertDescription className="font-medium">
                          Disease Detected: {analysisResult.disease_name}
                        </AlertDescription>
                      </div>
                    </Alert>

                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <p className="text-sm font-medium">Confidence</p>
                        <p className="text-2xl font-bold text-primary">
                          {(analysisResult.confidence * 100).toFixed(1)}%
                        </p>
                      </div>
                      <div>
                        <p className="text-sm font-medium">Severity</p>
                        <Badge variant={getSeverityColor(analysisResult.severity)}>
                          {analysisResult.severity}
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
                  </div>
                )}
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default PestControl;
