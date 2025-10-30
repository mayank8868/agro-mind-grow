import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { 
  Cloud, 
  TrendingUp, 
  Calendar, 
  Bug, 
  Wrench, 
  Users, 
  Map, 
  FileText, 
  BookOpen,
  Sprout,
  ArrowRight
} from "lucide-react";
import agricultureHero from "@/assets/agriculture-hero.jpg";

const Index = () => {
  const features = [
    {
      to: "/weather",
      icon: Cloud,
      title: "Weather & Climate",
      description: "Real-time weather data, forecasts, and climate analysis for better crop planning",
      color: "from-blue-500 to-cyan-500"
    },
    {
      to: "/market-prices",
      icon: TrendingUp,
      title: "Market Prices",
      description: "Live crop prices, market trends, and profit analysis across different regions",
      color: "from-green-500 to-emerald-500"
    },
    {
      to: "/crop-calendar",
      icon: Calendar,
      title: "Crop Calendar",
      description: "Seasonal planting guides, harvesting times, and agricultural calendar",
      color: "from-amber-500 to-orange-500"
    },
    {
      to: "/pest-control",
      icon: Bug,
      title: "Pest & Disease Control",
      description: "Identify pests, diseases, and get treatment recommendations",
      color: "from-red-500 to-pink-500"
    },
    {
      to: "/equipment",
      icon: Wrench,
      title: "Equipment Catalog",
      description: "Modern farming equipment, tools, and machinery information",
      color: "from-gray-500 to-slate-500"
    },
    {
      to: "/planning",
      icon: Users,
      title: "Planning & Consultation",
      description: "Farm planning and book consultations with agricultural experts",
      color: "from-purple-500 to-violet-500"
    },
    {
      to: "/knowledge-base",
      icon: BookOpen,
      title: "Knowledge Base & Schemes",
      description: "Learn farming techniques and access government schemes",
      color: "from-emerald-500 to-green-500"
    }
  ];

  return (
    <div className="min-h-screen bg-background">
      
      {/* Hero Section */}
      <section className="relative py-20 overflow-hidden">
        <div 
          className="absolute inset-0 bg-cover bg-center bg-no-repeat"
          style={{ backgroundImage: `url(${agricultureHero})` }}
        />
        <div className="absolute inset-0 bg-gradient-to-r from-primary/90 to-primary/70" />
        
        <div className="relative w-full px-4 text-center">
          <div className="max-w-4xl mx-auto">
            <h1 className="text-5xl md:text-7xl font-bold text-white mb-6">
              Smart Agriculture
              <span className="block bg-gradient-to-r from-yellow-300 to-green-300 bg-clip-text text-transparent">
                Platform
              </span>
            </h1>
            <p className="text-xl md:text-2xl text-white/90 mb-8 leading-relaxed">
              Empowering farmers with AI-driven insights, real-time data, and expert guidance 
              for sustainable and profitable farming
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link to="/crop-calendar">
                <Button size="lg" className="bg-white text-primary hover:bg-white/90 shadow-lg">
                  <Sprout className="mr-2 h-5 w-5" />
                  Start Farming Smart
                </Button>
              </Link>
              <Link to="/planning">
                <Button size="lg" variant="outline" className="border-white text-white hover:bg-white hover:text-primary bg-transparent backdrop-blur-sm">
                  Consult Experts
                  <ArrowRight className="ml-2 h-5 w-5" />
                </Button>
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Features Grid */}
      <section className="py-20 bg-gradient-earth">
        <div className="w-full px-4">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold mb-4">Comprehensive Farming Solutions</h2>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              Access all the tools and information you need to make data-driven farming decisions
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {features.map((feature) => {
              const Icon = feature.icon;
              return (
                <Link key={feature.to} to={feature.to} className="group">
                  <Card className="h-full border-0 shadow-soft hover:shadow-green transition-all duration-300 hover:scale-105">
                    <CardHeader className="text-center">
                      <div className={`w-16 h-16 mx-auto rounded-full bg-gradient-to-r ${feature.color} flex items-center justify-center shadow-lg`}>
                        <Icon className="h-8 w-8 text-white" />
                      </div>
                      <CardTitle className="text-xl group-hover:text-primary transition-colors">
                        {feature.title}
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <CardDescription className="text-center text-base leading-relaxed">
                        {feature.description}
                      </CardDescription>
                    </CardContent>
                  </Card>
                </Link>
              );
            })}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-16 bg-gradient-primary">
        <div className="w-full px-4 text-center">
          <h3 className="text-3xl font-bold text-white mb-4">
            Ready to Transform Your Farming?
          </h3>
          <p className="text-white/90 text-lg mb-8 max-w-2xl mx-auto">
            Join thousands of farmers who are already using smart agriculture techniques 
            to increase their yield and profits
          </p>
          <Link to="/planning">
            <Button size="lg" className="bg-white text-primary hover:bg-white/90">
              Get Started Today
              <ArrowRight className="ml-2 h-5 w-5" />
            </Button>
          </Link>
        </div>
      </section>
    </div>
  );
};

export default Index;