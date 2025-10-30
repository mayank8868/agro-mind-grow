import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { BookOpen, Video, FileText, Users, Search, Clock, ThumbsUp, Eye, ExternalLink, CheckCircle, Calendar, DollarSign } from "lucide-react";

const KnowledgeBase = () => {
  const [selectedTab, setSelectedTab] = useState("knowledge");

  // Knowledge Base Categories
  const categories = [
    { name: "Crop Management", articles: 145, icon: "🌾", description: "Learn about different crops and their management" },
    { name: "Soil & Fertilizers", articles: 89, icon: "🌱", description: "Soil health and fertilization techniques" },
    { name: "Pest Control", articles: 67, icon: "🐛", description: "Integrated pest management strategies" },
    { name: "Water Management", articles: 54, icon: "💧", description: "Irrigation and water conservation methods" },
    { name: "Equipment & Tools", articles: 78, icon: "🚜", description: "Farming equipment and modern tools" },
    { name: "Organic Farming", articles: 56, icon: "🌿", description: "Sustainable and organic farming practices" },
    { name: "Weather & Climate", articles: 34, icon: "🌤️", description: "Weather patterns and climate adaptation" },
    { name: "Market & Finance", articles: 43, icon: "💰", description: "Marketing strategies and financial planning" }
  ];

  // Featured Articles
  const featuredArticles = [
    {
      title: "Complete Guide to Rice Cultivation",
      category: "Crop Management",
      author: "Dr. Rajesh Kumar",
      readTime: "15 min read",
      views: 12500,
      likes: 890,
      publishDate: "March 10, 2024",
      description: "Comprehensive guide covering all aspects of rice cultivation from seed selection to harvesting",
      type: "Article",
      difficulty: "Beginner"
    },
    {
      title: "Drip Irrigation System Setup",
      category: "Water Management", 
      author: "Prof. Anil Verma",
      readTime: "25 min watch",
      views: 8900,
      likes: 654,
      publishDate: "March 8, 2024",
      description: "Step-by-step video guide on setting up efficient drip irrigation systems",
      type: "Video",
      difficulty: "Intermediate"
    },
    {
      title: "Organic Pest Control Methods",
      category: "Pest Control",
      author: "Dr. Priya Sharma",
      readTime: "12 min read",
      views: 7800,
      likes: 542,
      publishDate: "March 5, 2024",
      description: "Natural and organic methods to control common agricultural pests",
      type: "Article",
      difficulty: "Beginner"
    },
    {
      title: "Soil pH Testing and Management",
      category: "Soil & Fertilizers",
      author: "Dr. Suresh Patel",
      readTime: "18 min read",
      views: 6500,
      likes: 423,
      publishDate: "March 3, 2024",
      description: "Understanding soil pH and how to manage it for optimal crop growth",
      type: "Article",
      difficulty: "Intermediate"
    }
  ];

  // Government Schemes Data
  const schemes = [
    {
      name: "PM-KISAN",
      fullName: "Pradhan Mantri Kisan Samman Nidhi",
      category: "Direct Benefit Transfer",
      amount: "₹6,000 per year",
      beneficiaries: "All farmers",
      status: "Active",
      deadline: "Ongoing",
      description: "Financial support to small and marginal farmers",
      eligibility: ["Land holding farmers", "Valid Aadhaar card", "Bank account linked"],
      documents: ["Aadhaar Card", "Bank Details", "Land Records"],
      applicationProcess: "Online through PM-KISAN portal"
    },
    {
      name: "PMFBY",
      fullName: "Pradhan Mantri Fasal Bima Yojana",
      category: "Crop Insurance",
      amount: "Up to ₹2 Lakh coverage",
      beneficiaries: "All farmers",
      status: "Active",
      deadline: "15 days after sowing",
      description: "Crop insurance scheme against natural calamities",
      eligibility: ["All farmers", "Notified crops", "Timely enrollment"],
      documents: ["Land Records", "Aadhaar Card", "Bank Details", "Sowing Certificate"],
      applicationProcess: "Through banks or insurance companies"
    },
    {
      name: "NABARD Subsidy",
      fullName: "NABARD Equipment Subsidy Scheme",
      category: "Equipment Subsidy",
      amount: "40-50% subsidy",
      beneficiaries: "Small & Marginal Farmers",
      status: "Active", 
      deadline: "March 31, 2024",
      description: "Subsidy on purchase of agricultural equipment",
      eligibility: ["Small/Marginal farmers", "First-time buyers", "Valid land records"],
      documents: ["Land Records", "Income Certificate", "Equipment Quotation"],
      applicationProcess: "Through NABARD offices or banks"
    },
    {
      name: "Soil Health Card",
      fullName: "Soil Health Card Scheme",
      category: "Soil Testing",
      amount: "Free soil testing",
      beneficiaries: "All farmers",
      status: "Active",
      deadline: "Ongoing",
      description: "Free soil testing and nutrient management advice",
      eligibility: ["Any farmer", "Land holding certificate"],
      documents: ["Land Records", "Application Form"],
      applicationProcess: "Through agriculture department"
    },
    {
      name: "PKVY",
      fullName: "Paramparagat Krishi Vikas Yojana",
      category: "Organic Farming",
      amount: "₹50,000 per hectare",
      beneficiaries: "Farmers adopting organic farming",
      status: "Active",
      deadline: "December 31, 2024",
      description: "Support for organic farming practices",
      eligibility: ["Group of farmers", "Organic farming commitment", "3-year cycle"],
      documents: ["Group Formation Certificate", "Land Records", "Organic Farming Plan"],
      applicationProcess: "Through agriculture department"
    }
  ];

  const applicationStatus = [
    {
      scheme: "PM-KISAN",
      applicationId: "PMKISAN2024001234",
      status: "Approved",
      appliedDate: "Jan 15, 2024",
      amount: "₹2,000",
      disbursementDate: "Feb 1, 2024"
    },
    {
      scheme: "PMFBY",
      applicationId: "PMFBY2024005678",
      status: "Under Review",
      appliedDate: "Mar 10, 2024",
      amount: "₹50,000",
      disbursementDate: "Pending"
    }
  ];

  const getTypeIcon = (type: string) => {
    switch (type.toLowerCase()) {
      case 'video':
        return <Video className="h-4 w-4" />;
      case 'article':
        return <FileText className="h-4 w-4" />;
      default:
        return <BookOpen className="h-4 w-4" />;
    }
  };

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty.toLowerCase()) {
      case 'beginner':
        return 'bg-green-100 text-green-800';
      case 'intermediate':
        return 'bg-yellow-100 text-yellow-800';
      case 'advanced':
        return 'bg-red-100 text-red-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'approved':
        return 'default';
      case 'under review':
        return 'secondary';
      case 'rejected':
        return 'destructive';
      default:
        return 'secondary';
    }
  };

  const getCategoryColor = (category: string) => {
    switch (category.toLowerCase()) {
      case 'direct benefit transfer':
        return 'bg-green-100 text-green-800';
      case 'crop insurance':
        return 'bg-blue-100 text-blue-800';
      case 'equipment subsidy':
        return 'bg-purple-100 text-purple-800';
      case 'soil testing':
        return 'bg-orange-100 text-orange-800';
      case 'organic farming':
        return 'bg-emerald-100 text-emerald-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div className="min-h-screen bg-background">
      <div className="w-full px-4 py-8">
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-2">Knowledge Base & Schemes</h1>
          <p className="text-muted-foreground text-lg">
            Learn farming techniques and access government schemes
          </p>
        </div>

        <Tabs value={selectedTab} onValueChange={setSelectedTab} className="w-full">
          <TabsList className="grid w-full grid-cols-2 mb-6">
            <TabsTrigger value="knowledge">Knowledge Base</TabsTrigger>
            <TabsTrigger value="schemes">Government Schemes</TabsTrigger>
          </TabsList>

          {/* Knowledge Base Tab */}
          <TabsContent value="knowledge" className="space-y-8">
            {/* Search */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Search className="h-5 w-5" />
                  Search Knowledge Base
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex gap-4">
                  <Input placeholder="Search articles, videos, guides..." className="flex-1" />
                  <Button>Search</Button>
                </div>
              </CardContent>
            </Card>

            {/* Categories */}
            <Card>
              <CardHeader>
                <CardTitle>Browse by Category</CardTitle>
                <CardDescription>Explore farming knowledge organized by topics</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                  {categories.map((category, index) => (
                    <div key={index} className="p-4 border rounded-lg hover:bg-muted cursor-pointer transition-colors">
                      <div className="text-center">
                        <div className="text-3xl mb-2">{category.icon}</div>
                        <h3 className="font-semibold mb-1">{category.name}</h3>
                        <p className="text-sm text-muted-foreground mb-2">{category.description}</p>
                        <Badge variant="outline">{category.articles} articles</Badge>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Featured Articles */}
            <Card>
              <CardHeader>
                <CardTitle>Featured Articles & Videos</CardTitle>
                <CardDescription>Most popular and recent content from our experts</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid gap-6">
                  {featuredArticles.map((article, index) => (
                    <div key={index} className="border rounded-lg p-4">
                      <div className="flex items-start justify-between mb-3">
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-2">
                            <h3 className="font-semibold text-lg">{article.title}</h3>
                            <Badge variant="outline" className="flex items-center gap-1">
                              {getTypeIcon(article.type)}
                              {article.type}
                            </Badge>
                          </div>
                          <p className="text-sm text-muted-foreground mb-2">{article.description}</p>
                          <div className="flex items-center gap-4 text-sm text-muted-foreground">
                            <span>By {article.author}</span>
                            <span>•</span>
                            <span className="flex items-center gap-1">
                              <Clock className="h-3 w-3" />
                              {article.readTime}
                            </span>
                            <span>•</span>
                            <span>{article.publishDate}</span>
                          </div>
                        </div>
                        <div className="text-right">
                          <Badge className={getDifficultyColor(article.difficulty)}>
                            {article.difficulty}
                          </Badge>
                          <div className="flex items-center gap-4 mt-2 text-sm text-muted-foreground">
                            <span className="flex items-center gap-1">
                              <Eye className="h-3 w-3" />
                              {article.views.toLocaleString()}
                            </span>
                            <span className="flex items-center gap-1">
                              <ThumbsUp className="h-3 w-3" />
                              {article.likes}
                            </span>
                          </div>
                        </div>
                      </div>
                      <div className="flex items-center justify-between">
                        <Badge variant="secondary">{article.category}</Badge>
                        <Button variant="outline" size="sm">Read {article.type}</Button>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Government Schemes Tab */}
          <TabsContent value="schemes" className="space-y-8">
            {/* My Applications */}
            <Card>
              <CardHeader>
                <CardTitle>My Applications</CardTitle>
                <CardDescription>Track your scheme application status</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {applicationStatus.map((app, index) => (
                    <div key={index} className="flex items-center justify-between p-4 border rounded-lg">
                      <div>
                        <h3 className="font-semibold">{app.scheme}</h3>
                        <p className="text-sm text-muted-foreground">Application ID: {app.applicationId}</p>
                        <p className="text-sm text-muted-foreground">Applied: {app.appliedDate}</p>
                      </div>
                      <div className="text-right">
                        <Badge variant={getStatusColor(app.status)}>{app.status}</Badge>
                        <div className="text-sm font-semibold mt-1">{app.amount}</div>
                        <div className="text-xs text-muted-foreground">
                          {app.disbursementDate !== 'Pending' ? `Disbursed: ${app.disbursementDate}` : 'Pending'}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Available Schemes */}
            <Card>
              <CardHeader>
                <CardTitle>Available Schemes</CardTitle>
                <CardDescription>Government schemes and subsidies for farmers</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-6">
                  {schemes.map((scheme, index) => (
                    <div key={index} className="border rounded-lg p-6">
                      <div className="flex items-start justify-between mb-4">
                        <div>
                          <div className="flex items-center gap-2 mb-2">
                            <h3 className="font-semibold text-xl">{scheme.name}</h3>
                            <Badge className={getCategoryColor(scheme.category)}>
                              {scheme.category}
                            </Badge>
                          </div>
                          <h4 className="text-muted-foreground mb-2">{scheme.fullName}</h4>
                          <p className="text-sm text-muted-foreground">{scheme.description}</p>
                        </div>
                        <div className="text-right">
                          <div className="text-lg font-bold text-primary">{scheme.amount}</div>
                          <div className="text-sm text-muted-foreground">for {scheme.beneficiaries}</div>
                        </div>
                      </div>

                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                        <div>
                          <h4 className="font-medium text-sm mb-2 flex items-center gap-1">
                            <CheckCircle className="h-3 w-3" />
                            Eligibility Criteria:
                          </h4>
                          <ul className="text-sm text-muted-foreground space-y-1">
                            {scheme.eligibility.map((criteria, idx) => (
                              <li key={idx} className="flex items-center gap-2">
                                <div className="w-1.5 h-1.5 bg-primary rounded-full"></div>
                                {criteria}
                              </li>
                            ))}
                          </ul>
                        </div>
                        
                        <div>
                          <h4 className="font-medium text-sm mb-2 flex items-center gap-1">
                            <FileText className="h-3 w-3" />
                            Required Documents:
                          </h4>
                          <ul className="text-sm text-muted-foreground space-y-1">
                            {scheme.documents.map((doc, idx) => (
                              <li key={idx} className="flex items-center gap-2">
                                <div className="w-1.5 h-1.5 bg-primary rounded-full"></div>
                                {doc}
                              </li>
                            ))}
                          </ul>
                        </div>
                        
                        <div>
                          <h4 className="font-medium text-sm mb-2 flex items-center gap-1">
                            <Calendar className="h-3 w-3" />
                            Application Details:
                          </h4>
                          <div className="space-y-2">
                            <div className="text-sm">
                              <span className="text-muted-foreground">Deadline: </span>
                              <span className="font-medium">{scheme.deadline}</span>
                            </div>
                            <div className="text-sm">
                              <span className="text-muted-foreground">Apply through: </span>
                              <span className="font-medium">{scheme.applicationProcess}</span>
                            </div>
                          </div>
                        </div>
                      </div>

                      <div className="flex items-center justify-between">
                        <Badge variant={scheme.status === 'Active' ? 'default' : 'secondary'}>
                          {scheme.status}
                        </Badge>
                        <div className="flex gap-2">
                          <Button variant="outline" size="sm">
                            <ExternalLink className="mr-2 h-4 w-4" />
                            View Details
                          </Button>
                          <Button size="sm">Apply Now</Button>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Quick Stats */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Total Schemes</CardTitle>
                  <FileText className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">156</div>
                  <p className="text-xs text-muted-foreground">Active schemes available</p>
                </CardContent>
              </Card>
              
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Total Beneficiaries</CardTitle>
                  <Users className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">2.3M</div>
                  <p className="text-xs text-muted-foreground">Farmers benefited</p>
                </CardContent>
              </Card>
              
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Amount Disbursed</CardTitle>
                  <DollarSign className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">₹45.2B</div>
                  <p className="text-xs text-muted-foreground">In current fiscal year</p>
                </CardContent>
              </Card>
              
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">My Applications</CardTitle>
                  <Calendar className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">3</div>
                  <p className="text-xs text-muted-foreground">Active applications</p>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};

export default KnowledgeBase;
