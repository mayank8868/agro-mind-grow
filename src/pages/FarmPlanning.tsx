import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Map, Calculator, BarChart3, Droplets, DollarSign, Calendar, PlusCircle } from "lucide-react";

const FarmPlanning = () => {
  const planningTools = [
    {
      title: "Crop Rotation Planner",
      description: "Plan optimal crop rotation to maintain soil health and maximize yield",
      icon: Calendar,
      features: ["Soil health optimization", "Pest reduction", "Yield maximization"],
      action: "Plan Rotation"
    },
    {
      title: "Field Layout Designer",
      description: "Design your farm layout for optimal space utilization",
      icon: Map,
      features: ["GPS mapping", "Area calculation", "Irrigation planning"],
      action: "Design Layout"
    },
    {
      title: "Resource Calculator",
      description: "Calculate seeds, fertilizers, and water requirements",
      icon: Calculator,
      features: ["Seed calculation", "Fertilizer planning", "Water requirements"],
      action: "Calculate Resources"
    },
    {
      title: "Profit Analyzer",
      description: "Analyze expected costs and profits for different crops",
      icon: BarChart3,
      features: ["Cost analysis", "Profit estimation", "ROI calculation"],
      action: "Analyze Profit"
    }
  ];

  const sampleFarmPlans = [
    {
      name: "5 Acre Mixed Farming Plan",
      crops: ["Rice (2 acres)", "Wheat (2 acres)", "Vegetables (1 acre)"],
      season: "Kharif-Rabi",
      expectedProfit: "₹2,50,000",
      waterRequirement: "High",
      investmentRequired: "₹80,000"
    },
    {
      name: "10 Acre Cash Crop Plan",
      crops: ["Cotton (6 acres)", "Sugarcane (4 acres)"],
      season: "Annual",
      expectedProfit: "₹6,00,000",
      waterRequirement: "Very High",
      investmentRequired: "₹2,00,000"
    },
    {
      name: "3 Acre Organic Plan",
      crops: ["Organic Rice (1.5 acres)", "Pulses (1.5 acres)"],
      season: "Kharif-Rabi",
      expectedProfit: "₹1,20,000",
      waterRequirement: "Medium",
      investmentRequired: "₹45,000"
    }
  ];

  const cropRotationExample = [
    { year: "Year 1", season1: "Rice", season2: "Wheat", season3: "Fallow" },
    { year: "Year 2", season1: "Cotton", season2: "Mustard", season3: "Green Manure" },
    { year: "Year 3", season1: "Pulses", season2: "Barley", season3: "Fodder" },
    { year: "Year 4", season1: "Rice", season2: "Wheat", season3: "Fallow" }
  ];

  return (
    <div className="min-h-screen bg-background">
      
      <div className="w-full px-4 py-8">
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-2">Farm Planning Tools</h1>
          <p className="text-muted-foreground text-lg">
            Plan your farm layout, crop rotation, and resource management efficiently
          </p>
        </div>

        {/* Farm Details Input */}
        <Card className="mb-8">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <PlusCircle className="h-5 w-5" />
              Create New Farm Plan
            </CardTitle>
            <CardDescription>Enter your farm details to get personalized planning recommendations</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <Input placeholder="Farm name" />
              <Input placeholder="Total area (acres)" type="number" />
              <Select>
                <SelectTrigger>
                  <SelectValue placeholder="Soil type" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="clay">Clay</SelectItem>
                  <SelectItem value="loamy">Loamy</SelectItem>
                  <SelectItem value="sandy">Sandy</SelectItem>
                  <SelectItem value="black">Black Soil</SelectItem>
                </SelectContent>
              </Select>
              <Select>
                <SelectTrigger>
                  <SelectValue placeholder="Water source" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="borewell">Borewell</SelectItem>
                  <SelectItem value="canal">Canal</SelectItem>
                  <SelectItem value="river">River</SelectItem>
                  <SelectItem value="rainwater">Rainwater</SelectItem>
                </SelectContent>
              </Select>
              <Input placeholder="Budget (₹)" type="number" />
              <Button>Generate Plan</Button>
            </div>
          </CardContent>
        </Card>

        {/* Planning Tools */}
        <Card className="mb-8">
          <CardHeader>
            <CardTitle>Farm Planning Tools</CardTitle>
            <CardDescription>Use these tools to optimize your farm operations</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {planningTools.map((tool, index) => {
                const Icon = tool.icon;
                return (
                  <div key={index} className="border rounded-lg p-4">
                    <div className="flex items-start gap-3 mb-3">
                      <Icon className="h-8 w-8 text-primary" />
                      <div>
                        <h3 className="font-semibold text-lg">{tool.title}</h3>
                        <p className="text-sm text-muted-foreground">{tool.description}</p>
                      </div>
                    </div>
                    <div className="space-y-2 mb-4">
                      {tool.features.map((feature, idx) => (
                        <div key={idx} className="flex items-center gap-2 text-sm">
                          <div className="w-1.5 h-1.5 bg-primary rounded-full"></div>
                          {feature}
                        </div>
                      ))}
                    </div>
                    <Button variant="outline" className="w-full">{tool.action}</Button>
                  </div>
                );
              })}
            </div>
          </CardContent>
        </Card>

        {/* Sample Farm Plans */}
        <Card className="mb-8">
          <CardHeader>
            <CardTitle>Sample Farm Plans</CardTitle>
            <CardDescription>Pre-designed farm plans for different farm sizes and crop combinations</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-4">
              {sampleFarmPlans.map((plan, index) => (
                <div key={index} className="border rounded-lg p-4">
                  <div className="flex items-start justify-between mb-4">
                    <div>
                      <h3 className="font-semibold text-lg">{plan.name}</h3>
                      <Badge variant="outline">{plan.season}</Badge>
                    </div>
                    <div className="text-right">
                      <div className="text-lg font-bold text-green-600">{plan.expectedProfit}</div>
                      <div className="text-sm text-muted-foreground">Expected Profit</div>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-4">
                    <div>
                      <h4 className="font-medium text-sm mb-1">Crops:</h4>
                      <div className="space-y-1">
                        {plan.crops.map((crop, idx) => (
                          <Badge key={idx} variant="secondary" className="block text-xs">{crop}</Badge>
                        ))}
                      </div>
                    </div>
                    <div>
                      <h4 className="font-medium text-sm mb-1">Investment:</h4>
                      <p className="text-sm font-semibold">{plan.investmentRequired}</p>
                    </div>
                    <div>
                      <h4 className="font-medium text-sm mb-1">Water Requirement:</h4>
                      <Badge variant={plan.waterRequirement === 'High' || plan.waterRequirement === 'Very High' ? 'destructive' : 'default'}>
                        {plan.waterRequirement}
                      </Badge>
                    </div>
                    <div className="flex items-end">
                      <Button size="sm" className="w-full">Use This Plan</Button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Crop Rotation Planner */}
        <Card className="mb-8">
          <CardHeader>
            <CardTitle>4-Year Crop Rotation Example</CardTitle>
            <CardDescription>Sustainable crop rotation plan for soil health and pest management</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full border-collapse">
                <thead>
                  <tr className="border-b">
                    <th className="text-left p-3 font-semibold">Year</th>
                    <th className="text-left p-3 font-semibold">Kharif Season</th>
                    <th className="text-left p-3 font-semibold">Rabi Season</th>
                    <th className="text-left p-3 font-semibold">Zaid Season</th>
                  </tr>
                </thead>
                <tbody>
                  {cropRotationExample.map((rotation, index) => (
                    <tr key={index} className="border-b hover:bg-muted/50">
                      <td className="p-3 font-medium">{rotation.year}</td>
                      <td className="p-3">
                        <Badge variant="default">{rotation.season1}</Badge>
                      </td>
                      <td className="p-3">
                        <Badge variant="secondary">{rotation.season2}</Badge>
                      </td>
                      <td className="p-3">
                        <Badge variant="outline">{rotation.season3}</Badge>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>

        {/* Resource Management */}
        <Card>
          <CardHeader>
            <CardTitle>Resource Management Dashboard</CardTitle>
            <CardDescription>Monitor and plan your farm resources</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="p-4 bg-blue-50 rounded-lg">
                <div className="flex items-center gap-2 mb-2">
                  <Droplets className="h-5 w-5 text-blue-600" />
                  <h4 className="font-semibold text-blue-800">Water Management</h4>
                </div>
                <p className="text-2xl font-bold text-blue-800">15,000L</p>
                <p className="text-sm text-blue-600">Daily requirement</p>
              </div>

              <div className="p-4 bg-green-50 rounded-lg">
                <div className="flex items-center gap-2 mb-2">
                  <DollarSign className="h-5 w-5 text-green-600" />
                  <h4 className="font-semibold text-green-800">Budget Tracking</h4>
                </div>
                <p className="text-2xl font-bold text-green-800">₹45,000</p>
                <p className="text-sm text-green-600">Remaining budget</p>
              </div>

              <div className="p-4 bg-orange-50 rounded-lg">
                <div className="flex items-center gap-2 mb-2">
                  <BarChart3 className="h-5 w-5 text-orange-600" />
                  <h4 className="font-semibold text-orange-800">Yield Forecast</h4>
                </div>
                <p className="text-2xl font-bold text-orange-800">25 Quintals</p>
                <p className="text-sm text-orange-600">Expected yield</p>
              </div>

              <div className="p-4 bg-purple-50 rounded-lg">
                <div className="flex items-center gap-2 mb-2">
                  <Calendar className="h-5 w-5 text-purple-600" />
                  <h4 className="font-semibold text-purple-800">Next Activity</h4>
                </div>
                <p className="text-lg font-bold text-purple-800">Irrigation</p>
                <p className="text-sm text-purple-600">In 2 days</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default FarmPlanning;