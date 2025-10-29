import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { FileText, Users, DollarSign, Calendar, Search, ExternalLink, CheckCircle } from "lucide-react";

const GovernmentSchemes = () => {
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
          <h1 className="text-4xl font-bold mb-2">Government Schemes</h1>
          <p className="text-muted-foreground text-lg">
            Access government subsidies, loans, and agricultural schemes
          </p>
        </div>

        {/* Search and Filter */}
        <Card className="mb-8">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Search className="h-5 w-5" />
              Find Schemes
            </CardTitle>
            <CardDescription>Search for government schemes based on your requirements</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <Input placeholder="Search schemes..." className="md:col-span-2" />
              <Select defaultValue="all-categories">
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all-categories">All Categories</SelectItem>
                  <SelectItem value="subsidies">Subsidies</SelectItem>
                  <SelectItem value="insurance">Insurance</SelectItem>
                  <SelectItem value="loans">Loans</SelectItem>
                  <SelectItem value="direct-benefit">Direct Benefit</SelectItem>
                </SelectContent>
              </Select>
              <Button>Search</Button>
            </div>
          </CardContent>
        </Card>

        {/* Application Status */}
        <Card className="mb-8">
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
        <Card className="mb-8">
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
      </div>
    </div>
  );
};

export default GovernmentSchemes;