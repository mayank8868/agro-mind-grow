import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useToast } from "@/hooks/use-toast";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { 
  Upload, 
  X, 
  Search, 
  Leaf, 
  AlertTriangle, 
  CheckCircle2, 
  Info, 
  Loader2, 
  Sprout,
  Bug,
  ShieldCheck
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

interface Prediction {
  class: string;
  confidence: number;
}

interface DiseaseResult {
  class: string;
  confidence: number;
  message: string;
  symptoms: string[];
  causes: string[];
  treatments: {
    chemical?: string[];
    organic?: string[];
    prevention?: string[];
  };
  prevention: string[];
  top3_predictions: Prediction[];
}

const PLANT_TYPES = [
  { value: "tomato", label: "Tomato" },
  { value: "potato", label: "Potato" },
  { value: "corn", label: "Corn (Maize)" },
  { value: "grape", label: "Grape" },
  { value: "apple", label: "Apple" },
  { value: "pepper", label: "Bell Pepper" },
  { value: "strawberry", label: "Strawberry" },
  { value: "peach", label: "Peach" },
  { value: "cherry", label: "Cherry" },
  { value: "blueberry", label: "Blueberry" },
  { value: "raspberry", label: "Raspberry" },
  { value: "soybean", label: "Soybean" },
  { value: "squash", label: "Squash" },
];

const PestControl = () => {
  const { toast } = useToast();
  const [selectedPlant, setSelectedPlant] = useState<string>("");
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<DiseaseResult | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Cleanup preview URL on unmount
  useEffect(() => {
    return () => {
      if (preview) URL.revokeObjectURL(preview);
    };
  }, [preview]);

  const handleFile = (selectedFile: File) => {
    if (!selectedFile.type.startsWith("image/")) {
      toast({
        title: "Invalid file type",
        description: "Please upload an image file (JPG, PNG, WEBP).",
        variant: "destructive",
      });
      return;
    }

    if (selectedFile.size > 10 * 1024 * 1024) {
      toast({
        title: "File too large",
        description: "Maximum file size is 10MB.",
        variant: "destructive",
      });
      return;
    }

    setFile(selectedFile);
    const objectUrl = URL.createObjectURL(selectedFile);
    setPreview(objectUrl);
    setResult(null);
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const clearSelection = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const analyzeImage = async () => {
    if (!file) return;

    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);
    if (selectedPlant) {
      formData.append("plant_type", selectedPlant);
    }

    try {
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Failed to analyze image");
      }

      if (data.class === "invalid_image") {
        toast({
          title: "Invalid Image Detected",
          description: data.message,
          variant: "destructive",
        });
        setResult(null);
      } else {
        setResult(data);
        toast({
          title: "Analysis Complete",
          description: "Here are the results for your plant.",
        });
      }
    } catch (error) {
      console.error("Analysis error:", error);
      toast({
        title: "Error",
        description: "Could not connect to the analysis server. Please try again.",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-green-50 to-white dark:from-green-950/20 dark:to-background p-4 md:p-8">
      <div className="max-w-5xl mx-auto space-y-8">
        
        {/* Header */}
        <div className="text-center space-y-4">
          <h1 className="text-4xl md:text-5xl font-bold text-green-800 dark:text-green-400 tracking-tight">
            Plant Doctor AI
          </h1>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            Upload a photo of your plant to instantly identify diseases and get expert treatment recommendations.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          
          {/* Left Column: Upload & Controls */}
          <div className="lg:col-span-1 space-y-6">
            <Card className="border-green-100 shadow-lg">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Search className="w-5 h-5 text-green-600" />
                  Analysis Setup
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                
                {/* Plant Type Selector */}
                <div className="space-y-2">
                  <label className="text-sm font-medium text-muted-foreground">
                    Plant Type (Optional)
                  </label>
                  <Select value={selectedPlant} onValueChange={setSelectedPlant}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select a plant..." />
                    </SelectTrigger>
                    <SelectContent>
                      {PLANT_TYPES.map((plant) => (
                        <SelectItem key={plant.value} value={plant.value}>
                          {plant.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <p className="text-xs text-muted-foreground">
                    Helps improve accuracy if you know the plant type.
                  </p>
                </div>

                {/* Upload Area */}
                <div
                  className={`relative border-2 border-dashed rounded-xl p-6 transition-all duration-200 ease-in-out text-center cursor-pointer
                    ${dragActive ? "border-green-500 bg-green-50 dark:bg-green-900/20" : "border-muted-foreground/25 hover:border-green-400 hover:bg-accent/50"}
                    ${preview ? "border-solid border-green-200 bg-green-50/50" : ""}
                  `}
                  onDragEnter={handleDrag}
                  onDragLeave={handleDrag}
                  onDragOver={handleDrag}
                  onDrop={handleDrop}
                  onClick={() => !preview && fileInputRef.current?.click()}
                >
                  <input
                    ref={fileInputRef}
                    type="file"
                    className="hidden"
                    accept="image/*"
                    onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
                  />

                  <AnimatePresence mode="wait">
                    {preview ? (
                      <motion.div
                        initial={{ opacity: 0, scale: 0.9 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0, scale: 0.9 }}
                        className="relative"
                      >
                        <img
                          src={preview}
                          alt="Preview"
                          className="w-full h-64 object-cover rounded-lg shadow-sm"
                        />
                        <Button
                          variant="destructive"
                          size="icon"
                          className="absolute -top-2 -right-2 h-8 w-8 rounded-full shadow-md"
                          onClick={(e) => {
                            e.stopPropagation();
                            clearSelection();
                          }}
                        >
                          <X className="w-4 h-4" />
                        </Button>
                      </motion.div>
                    ) : (
                      <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="py-8 space-y-4"
                      >
                        <div className="w-16 h-16 bg-green-100 dark:bg-green-900/50 rounded-full flex items-center justify-center mx-auto">
                          <Upload className="w-8 h-8 text-green-600 dark:text-green-400" />
                        </div>
                        <div>
                          <p className="font-medium text-foreground">
                            Click or drag image here
                          </p>
                          <p className="text-sm text-muted-foreground mt-1">
                            Supports JPG, PNG, WEBP
                          </p>
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>

                <Button
                  className="w-full bg-green-600 hover:bg-green-700 text-white"
                  size="lg"
                  disabled={!file || loading}
                  onClick={analyzeImage}
                >
                  {loading ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <Sprout className="w-4 h-4 mr-2" />
                      Diagnose Plant
                    </>
                  )}
                </Button>
              </CardContent>
            </Card>
          </div>

          {/* Right Column: Results */}
          <div className="lg:col-span-2">
            <AnimatePresence mode="wait">
              {result ? (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  className="space-y-6"
                >
                  {/* Main Diagnosis Card */}
                  <Card className={`border-l-4 shadow-md ${
                    result.class.toLowerCase().includes("healthy") 
                      ? "border-l-green-500" 
                      : "border-l-red-500"
                  }`}>
                    <CardHeader>
                      <div className="flex items-start justify-between">
                        <div>
                          <CardTitle className="text-2xl flex items-center gap-2">
                            {result.class.toLowerCase().includes("healthy") ? (
                              <CheckCircle2 className="w-6 h-6 text-green-500" />
                            ) : (
                              <AlertTriangle className="w-6 h-6 text-red-500" />
                            )}
                            {result.class.replace(/_/g, " ").replace("___", " - ")}
                          </CardTitle>
                          <CardDescription className="mt-2 text-base">
                            {result.message}
                          </CardDescription>
                        </div>
                        <Badge 
                          variant={result.confidence > 70 ? "default" : "secondary"}
                          className="text-lg px-3 py-1"
                        >
                          {result.confidence.toFixed(1)}% Confidence
                        </Badge>
                      </div>
                    </CardHeader>
                    
                    {/* Top 3 Predictions (if confidence is low/medium) */}
                    {result.confidence < 85 && (
                      <CardContent className="pb-2">
                        <div className="bg-muted/50 p-3 rounded-lg text-sm">
                          <p className="font-medium mb-2 text-muted-foreground">Alternative Possibilities:</p>
                          <div className="space-y-1">
                            {result.top3_predictions.slice(1).map((pred, idx) => (
                              <div key={idx} className="flex justify-between items-center">
                                <span>{pred.class.replace(/_/g, " ")}</span>
                                <span className="font-mono text-muted-foreground">{pred.confidence.toFixed(1)}%</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      </CardContent>
                    )}
                  </Card>

                  {/* Detailed Info Grid */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    
                    {/* Symptoms */}
                    <Card>
                      <CardHeader>
                        <CardTitle className="flex items-center gap-2 text-lg">
                          <Info className="w-5 h-5 text-blue-500" />
                          Symptoms
                        </CardTitle>
                      </CardHeader>
                      <CardContent>
                        <ul className="space-y-2">
                          {result.symptoms.length > 0 ? (
                            result.symptoms.map((symptom, i) => (
                              <li key={i} className="flex items-start gap-2 text-sm">
                                <span className="mt-1.5 w-1.5 h-1.5 bg-blue-400 rounded-full flex-shrink-0" />
                                {symptom}
                              </li>
                            ))
                          ) : (
                            <li className="text-muted-foreground text-sm">No specific symptoms listed.</li>
                          )}
                        </ul>
                      </CardContent>
                    </Card>

                    {/* Causes */}
                    <Card>
                      <CardHeader>
                        <CardTitle className="flex items-center gap-2 text-lg">
                          <Bug className="w-5 h-5 text-orange-500" />
                          Causes
                        </CardTitle>
                      </CardHeader>
                      <CardContent>
                        <ul className="space-y-2">
                          {result.causes.length > 0 ? (
                            result.causes.map((cause, i) => (
                              <li key={i} className="flex items-start gap-2 text-sm">
                                <span className="mt-1.5 w-1.5 h-1.5 bg-orange-400 rounded-full flex-shrink-0" />
                                {cause}
                              </li>
                            ))
                          ) : (
                            <li className="text-muted-foreground text-sm">No specific causes listed.</li>
                          )}
                        </ul>
                      </CardContent>
                    </Card>

                    {/* Treatments */}
                    <Card className="md:col-span-2">
                      <CardHeader>
                        <CardTitle className="flex items-center gap-2 text-lg">
                          <ShieldCheck className="w-5 h-5 text-green-600" />
                          Treatment & Prevention
                        </CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                          
                          {/* Organic */}
                          <div>
                            <h4 className="font-semibold text-green-700 mb-2 flex items-center gap-2">
                              <Leaf className="w-4 h-4" /> Organic
                            </h4>
                            <ul className="space-y-2">
                              {result.treatments.organic?.map((t, i) => (
                                <li key={i} className="text-sm text-muted-foreground border-l-2 border-green-200 pl-2">
                                  {t}
                                </li>
                              )) || <li className="text-sm text-muted-foreground">No organic treatments listed.</li>}
                            </ul>
                          </div>

                          {/* Chemical */}
                          <div>
                            <h4 className="font-semibold text-purple-700 mb-2 flex items-center gap-2">
                              <AlertTriangle className="w-4 h-4" /> Chemical
                            </h4>
                            <ul className="space-y-2">
                              {result.treatments.chemical?.map((t, i) => (
                                <li key={i} className="text-sm text-muted-foreground border-l-2 border-purple-200 pl-2">
                                  {t}
                                </li>
                              )) || <li className="text-sm text-muted-foreground">No chemical treatments listed.</li>}
                            </ul>
                          </div>

                          {/* Prevention */}
                          <div>
                            <h4 className="font-semibold text-blue-700 mb-2 flex items-center gap-2">
                              <ShieldCheck className="w-4 h-4" /> Prevention
                            </h4>
                            <ul className="space-y-2">
                              {result.prevention?.map((p, i) => (
                                <li key={i} className="text-sm text-muted-foreground border-l-2 border-blue-200 pl-2">
                                  {p}
                                </li>
                              )) || <li className="text-sm text-muted-foreground">No prevention tips listed.</li>}
                            </ul>
                          </div>

                        </div>
                      </CardContent>
                    </Card>

                  </div>
                </motion.div>
              ) : (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="h-full flex flex-col items-center justify-center text-center p-12 border-2 border-dashed rounded-xl border-muted-foreground/10 bg-muted/5"
                >
                  <div className="w-24 h-24 bg-green-100 dark:bg-green-900/20 rounded-full flex items-center justify-center mb-6">
                    <Sprout className="w-12 h-12 text-green-600 dark:text-green-400" />
                  </div>
                  <h3 className="text-xl font-semibold mb-2">Ready to Analyze</h3>
                  <p className="text-muted-foreground max-w-md">
                    Upload a clear image of a plant leaf to detect diseases. 
                    Our AI model analyzes patterns to provide accurate diagnoses and treatment plans.
                  </p>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

        </div>
      </div>
    </div>
  );
};

export default PestControl;