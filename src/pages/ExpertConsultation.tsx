import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Calendar, Clock, Video, MessageCircle, Star, Award, MapPin } from "lucide-react";

const ExpertConsultation = () => {
  const experts = [
    {
      name: "Dr. Rajesh Kumar",
      title: "Senior Agricultural Scientist",
      specialization: ["Crop Protection", "Pest Management", "Organic Farming"],
      experience: "15+ years",
      rating: 4.9,
      reviews: 234,
      price: "₹500/consultation",
      languages: ["Hindi", "English", "Punjabi"],
      location: "Punjab Agricultural University",
      availability: "Available Today",
      image: "/placeholder.svg"
    },
    {
      name: "Dr. Priya Sharma",
      title: "Soil Health Expert",
      specialization: ["Soil Testing", "Nutrition Management", "Soil Conservation"],
      experience: "12+ years",
      rating: 4.8,
      reviews: 189,
      price: "₹400/consultation",
      languages: ["Hindi", "English"],
      location: "IARI, New Delhi",
      availability: "Available Tomorrow",
      image: "/placeholder.svg"
    },
    {
      name: "Prof. Anil Verma",
      title: "Irrigation Specialist",
      specialization: ["Water Management", "Drip Irrigation", "Farm Planning"],
      experience: "20+ years",
      rating: 4.9,
      reviews: 312,
      price: "₹600/consultation",
      languages: ["Hindi", "English", "Marathi"],
      location: "Maharashtra Agricultural University",
      availability: "Available in 2 days",
      image: "/placeholder.svg"
    }
  ];

  const consultationTypes = [
    {
      type: "Video Call",
      duration: "30-45 minutes",
      description: "One-on-one video consultation with screen sharing",
      icon: Video,
      features: ["Real-time discussion", "Document sharing", "Recording available"]
    },
    {
      type: "Chat Support",
      duration: "Instant responses",
      description: "Quick text-based consultation for immediate queries",
      icon: MessageCircle,
      features: ["Immediate responses", "Photo sharing", "Follow-up messages"]
    },
    {
      type: "Field Visit",
      duration: "2-3 hours",
      description: "Expert visits your farm for hands-on consultation",
      icon: MapPin,
      features: ["On-site inspection", "Detailed report", "Action plan"]
    }
  ];

  const recentConsultations = [
    {
      farmer: "Ramesh Singh",
      expert: "Dr. Rajesh Kumar",
      topic: "Rice blast disease treatment",
      date: "2 days ago",
      rating: 5,
      feedback: "Very helpful advice on managing rice blast. The organic treatment suggestions worked perfectly."
    },
    {
      farmer: "Sunita Devi",
      expert: "Dr. Priya Sharma", 
      topic: "Soil pH management",
      date: "5 days ago",
      rating: 4,
      feedback: "Detailed explanation about soil testing and pH correction methods. Will implement the suggestions."
    }
  ];

  return (
    <div className="min-h-screen bg-background">
      
      <div className="w-full px-4 py-8">
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-2">Expert Consultation</h1>
          <p className="text-muted-foreground text-lg">
            Connect with agricultural experts for personalized farming advice
          </p>
        </div>

        {/* Quick Consultation Request */}
        <Card className="mb-8">
          <CardHeader>
            <CardTitle>Quick Consultation Request</CardTitle>
            <CardDescription>Describe your farming challenge and get matched with the right expert</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-4">
                <Input placeholder="Your name" />
                <Input placeholder="Location (State, District)" />
                <Input placeholder="Crop type" />
                <Textarea 
                  placeholder="Describe your problem or question..."
                  rows={4}
                />
              </div>
              <div className="space-y-4">
                <div>
                  <h4 className="font-semibold mb-3">Preferred Consultation Type</h4>
                  <div className="space-y-2">
                    {consultationTypes.map((type, index) => {
                      const Icon = type.icon;
                      return (
                        <div key={index} className="flex items-center space-x-2">
                          <input type="radio" name="consultation-type" id={`type-${index}`} className="text-primary" />
                          <label htmlFor={`type-${index}`} className="flex items-center gap-2 cursor-pointer">
                            <Icon className="h-4 w-4" />
                            <span>{type.type}</span>
                            <span className="text-sm text-muted-foreground">({type.duration})</span>
                          </label>
                        </div>
                      );
                    })}
                  </div>
                </div>
                <Button className="w-full">
                  Find Expert & Book Consultation
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Consultation Types */}
        <Card className="mb-8">
          <CardHeader>
            <CardTitle>Consultation Options</CardTitle>
            <CardDescription>Choose the consultation format that works best for you</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {consultationTypes.map((type, index) => {
                const Icon = type.icon;
                return (
                  <div key={index} className="border rounded-lg p-4">
                    <div className="flex items-center gap-3 mb-3">
                      <Icon className="h-8 w-8 text-primary" />
                      <div>
                        <h3 className="font-semibold">{type.type}</h3>
                        <p className="text-sm text-muted-foreground">{type.duration}</p>
                      </div>
                    </div>
                    <p className="text-sm text-muted-foreground mb-4">{type.description}</p>
                    <div className="space-y-1">
                      {type.features.map((feature, idx) => (
                        <div key={idx} className="flex items-center gap-2 text-sm">
                          <div className="w-1.5 h-1.5 bg-primary rounded-full"></div>
                          {feature}
                        </div>
                      ))}
                    </div>
                  </div>
                );
              })}
            </div>
          </CardContent>
        </Card>

        {/* Available Experts */}
        <Card className="mb-8">
          <CardHeader>
            <CardTitle>Available Experts</CardTitle>
            <CardDescription>Connect with certified agricultural experts</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-6">
              {experts.map((expert, index) => (
                <div key={index} className="border rounded-lg p-6">
                  <div className="flex items-start gap-4">
                    <Avatar className="h-16 w-16">
                      <AvatarImage src={expert.image} alt={expert.name} />
                      <AvatarFallback>{expert.name.split(' ').map(n => n[0]).join('')}</AvatarFallback>
                    </Avatar>
                    
                    <div className="flex-1">
                      <div className="flex items-start justify-between mb-2">
                        <div>
                          <h3 className="font-semibold text-lg">{expert.name}</h3>
                          <p className="text-muted-foreground">{expert.title}</p>
                          <div className="flex items-center gap-2 mt-1">
                            <MapPin className="h-3 w-3 text-muted-foreground" />
                            <span className="text-sm text-muted-foreground">{expert.location}</span>
                          </div>
                        </div>
                        <div className="text-right">
                          <div className="flex items-center gap-1 mb-1">
                            <Star className="h-4 w-4 fill-yellow-400 text-yellow-400" />
                            <span className="font-semibold">{expert.rating}</span>
                            <span className="text-sm text-muted-foreground">({expert.reviews})</span>
                          </div>
                          <div className="text-lg font-bold text-primary">{expert.price}</div>
                        </div>
                      </div>

                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                        <div>
                          <h4 className="font-medium text-sm mb-2">Specialization:</h4>
                          <div className="flex flex-wrap gap-1">
                            {expert.specialization.map((spec, idx) => (
                              <Badge key={idx} variant="secondary">{spec}</Badge>
                            ))}
                          </div>
                        </div>
                        <div>
                          <h4 className="font-medium text-sm mb-2">Experience:</h4>
                          <div className="flex items-center gap-2">
                            <Award className="h-4 w-4 text-primary" />
                            <span>{expert.experience}</span>
                          </div>
                        </div>
                        <div>
                          <h4 className="font-medium text-sm mb-2">Languages:</h4>
                          <p className="text-sm">{expert.languages.join(', ')}</p>
                        </div>
                      </div>

                      <div className="flex items-center justify-between">
                        <Badge variant={expert.availability.includes('Today') ? 'default' : 'secondary'}>
                          <Clock className="h-3 w-3 mr-1" />
                          {expert.availability}
                        </Badge>
                        <div className="flex gap-2">
                          <Button variant="outline">View Profile</Button>
                          <Button>
                            <Calendar className="mr-2 h-4 w-4" />
                            Book Consultation
                          </Button>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Recent Consultations */}
        <Card>
          <CardHeader>
            <CardTitle>Recent Consultations</CardTitle>
            <CardDescription>Feedback from farmers who consulted our experts</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {recentConsultations.map((consultation, index) => (
                <div key={index} className="border rounded-lg p-4">
                  <div className="flex items-start justify-between mb-3">
                    <div>
                      <h4 className="font-semibold">{consultation.farmer}</h4>
                      <p className="text-sm text-muted-foreground">
                        Consulted with {consultation.expert} • {consultation.topic}
                      </p>
                    </div>
                    <div className="text-right">
                      <div className="flex items-center gap-1">
                        {[...Array(5)].map((_, i) => (
                          <Star
                            key={i}
                            className={`h-3 w-3 ${i < consultation.rating ? 'fill-yellow-400 text-yellow-400' : 'text-gray-300'}`}
                          />
                        ))}
                      </div>
                      <p className="text-xs text-muted-foreground">{consultation.date}</p>
                    </div>
                  </div>
                  <p className="text-sm text-muted-foreground italic">"{consultation.feedback}"</p>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default ExpertConsultation;