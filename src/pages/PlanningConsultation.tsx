import { useState, useEffect, useRef } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { useToast } from "@/hooks/use-toast";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger, DialogFooter } from "@/components/ui/dialog";
import { Textarea } from "@/components/ui/textarea";
import { Calendar as CalIcon, MapPin, Users, Clock, CheckCircle2, Mail, Phone, Video, Calendar } from "lucide-react";

const PlanningConsultation = () => {
  const { toast } = useToast();

  // Farm Details Form
  const [farmName, setFarmName] = useState("");
  const [area, setArea] = useState("");
  const [soil, setSoil] = useState("");
  const [water, setWater] = useState("");
  const [cropType, setCropType] = useState("");
  const [location, setLocation] = useState("");
  const [budget, setBudget] = useState("");
  const [farmType, setFarmType] = useState("");
  const [experienceLevel, setExperienceLevel] = useState("");

  // Booking dialog
  const [isBookingOpen, setIsBookingOpen] = useState(false);
  const [selectedExpert, setSelectedExpert] = useState<string>("");
  const [selectedDate, setSelectedDate] = useState("");
  const [selectedTime, setSelectedTime] = useState("");
  const [concern, setConcern] = useState("");
  const [bookingConfirmed, setBookingConfirmed] = useState(false);

  const experts = [
    { id: "1", name: "Dr. Rajesh Kumar", specialization: "Soil & Fertilization", experience: "15 years", rating: 4.8 },
    { id: "2", name: "Prof. Anil Verma", specialization: "Irrigation & Water Management", experience: "12 years", rating: 4.9 },
    { id: "3", name: "Dr. Priya Sharma", specialization: "Pest & Disease Control", experience: "10 years", rating: 4.7 },
    { id: "4", name: "Dr. Suresh Patel", specialization: "Crop Planning & Rotation", experience: "18 years", rating: 4.9 },
    { id: "5", name: "Dr. Meera Reddy", specialization: "Organic Farming", experience: "14 years", rating: 4.8 },
    { id: "6", name: "Dr. Amit Singh", specialization: "Farm Economics", experience: "11 years", rating: 4.6 },
  ];

  const timeSlots = [
    "09:00 AM", "10:00 AM", "11:00 AM", "12:00 PM",
    "02:00 PM", "03:00 PM", "04:00 PM", "05:00 PM"
  ];

  const [availableDates, setAvailableDates] = useState<string[]>([]);
  
  useEffect(() => {
    // Generate next 14 days as available dates
    const dates: string[] = [];
    const today = new Date();
    for (let i = 1; i <= 14; i++) {
      const date = new Date(today);
      date.setDate(today.getDate() + i);
      dates.push(date.toISOString().split('T')[0]);
    }
    setAvailableDates(dates);
  }, []);

  const handleBooking = () => {
    if (!selectedExpert || !selectedDate || !selectedTime || !concern.trim()) {
      toast({
        title: "Incomplete Information",
        description: "Please fill all fields in the booking form",
        variant: "destructive"
      });
      return;
    }

    const expertName = experts.find(e => e.id === selectedExpert)?.name;
    toast({
      title: "✅ Appointment Booked!",
      description: `You have scheduled a consultation with ${expertName} on ${new Date(selectedDate).toLocaleDateString()} at ${selectedTime}`
    });

    setBookingConfirmed(true);
    setIsBookingOpen(false);
    
    // Reset after 3 seconds
    setTimeout(() => {
      setBookingConfirmed(false);
      setSelectedExpert("");
      setSelectedDate("");
      setSelectedTime("");
      setConcern("");
    }, 3000);
  };

  return (
    <div className="min-h-screen bg-background">
      <div className="w-full px-4 py-8">
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-2">Farm Planning & Expert Consultation</h1>
          <p className="text-muted-foreground text-lg">
            Plan your farm activities and book consultations with agricultural experts
          </p>
        </div>

        {/* Farm Details Form */}
        <Card className="mb-8">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <MapPin className="h-5 w-5" /> Farm Details
            </CardTitle>
            <CardDescription>Enter your farm information for planning and consultations</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              <div>
                <label className="text-sm font-medium mb-1 block">Farm Name</label>
                <Input placeholder="Enter farm name" value={farmName} onChange={(e) => setFarmName(e.target.value)} />
              </div>
              <div>
                <label className="text-sm font-medium mb-1 block">Total Area (acres)</label>
                <Input type="number" placeholder="Enter area" value={area} onChange={(e) => setArea(e.target.value)} />
              </div>
              <div>
                <label className="text-sm font-medium mb-1 block">Soil Type</label>
                <Select value={soil} onValueChange={setSoil}>
                  <SelectTrigger><SelectValue placeholder="Select soil type" /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="loam">Loam</SelectItem>
                    <SelectItem value="clay">Clay</SelectItem>
                    <SelectItem value="sandy">Sandy</SelectItem>
                    <SelectItem value="red">Red Soil</SelectItem>
                    <SelectItem value="black">Black Soil</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div>
                <label className="text-sm font-medium mb-1 block">Water Source</label>
                <Select value={water} onValueChange={setWater}>
                  <SelectTrigger><SelectValue placeholder="Select water source" /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="canal">Canal</SelectItem>
                    <SelectItem value="well">Well/Borewell</SelectItem>
                    <SelectItem value="rainfed">Rainfed</SelectItem>
                    <SelectItem value="river">River</SelectItem>
                    <SelectItem value="pond">Pond/Tank</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div>
                <label className="text-sm font-medium mb-1 block">Crop Type</label>
                <Select value={cropType} onValueChange={setCropType}>
                  <SelectTrigger><SelectValue placeholder="Select crop type" /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="cereal">Cereals (Rice, Wheat, Maize)</SelectItem>
                    <SelectItem value="vegetable">Vegetables</SelectItem>
                    <SelectItem value="fruit">Fruits</SelectItem>
                    <SelectItem value="pulses">Pulses</SelectItem>
                    <SelectItem value="oilseeds">Oilseeds</SelectItem>
                    <SelectItem value="cash">Cash Crops</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div>
                <label className="text-sm font-medium mb-1 block">Location</label>
                <Input placeholder="City, State" value={location} onChange={(e) => setLocation(e.target.value)} />
              </div>
              <div>
                <label className="text-sm font-medium mb-1 block">Budget (₹)</label>
                <Input type="number" placeholder="Enter budget" value={budget} onChange={(e) => setBudget(e.target.value)} />
              </div>
              <div>
                <label className="text-sm font-medium mb-1 block">Farm Type</label>
                <Select value={farmType} onValueChange={setFarmType}>
                  <SelectTrigger><SelectValue placeholder="Select farm type" /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="organic">Organic Farming</SelectItem>
                    <SelectItem value="conventional">Conventional</SelectItem>
                    <SelectItem value="mixed">Mixed</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div>
                <label className="text-sm font-medium mb-1 block">Experience Level</label>
                <Select value={experienceLevel} onValueChange={setExperienceLevel}>
                  <SelectTrigger><SelectValue placeholder="Select level" /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="beginner">Beginner (0-2 years)</SelectItem>
                    <SelectItem value="intermediate">Intermediate (3-5 years)</SelectItem>
                    <SelectItem value="experienced">Experienced (6+ years)</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
            <div className="mt-6 flex justify-end">
              <Button size="lg">Save Farm Details</Button>
            </div>
          </CardContent>
        </Card>

        {/* Expert Consultation Section */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left: Available Experts */}
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Users className="h-5 w-5" /> Available Agricultural Experts
                </CardTitle>
                <CardDescription>Book a consultation with certified experts</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid gap-4">
                  {experts.map((expert) => (
                    <Card key={expert.id} className="p-4 hover:shadow-md transition-shadow">
                      <div className="flex items-start justify-between mb-3">
                        <div>
                          <h3 className="font-semibold text-lg">{expert.name}</h3>
                          <p className="text-sm text-muted-foreground">{expert.specialization}</p>
                          <div className="flex items-center gap-4 mt-2 text-xs text-muted-foreground">
                            <span className="flex items-center gap-1">
                              <Clock className="h-3 w-3" />
                              {expert.experience}
                            </span>
                            <span className="flex items-center gap-1">
                              <CheckCircle2 className="h-3 w-3" />
                              Rating: {expert.rating}/5
                            </span>
                          </div>
                        </div>
                        <Badge variant="secondary" className="ml-2">{expert.specialization}</Badge>
                      </div>
                      <Dialog open={isBookingOpen && selectedExpert === expert.id} onOpenChange={setIsBookingOpen}>
                        <DialogTrigger asChild>
                          <Button 
                            variant="default" 
                            className="w-full"
                            onClick={() => setSelectedExpert(expert.id)}
                          >
                            Book Consultation
                          </Button>
                        </DialogTrigger>
                        <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
                          <DialogHeader>
                            <DialogTitle>Book Consultation with {expert.name}</DialogTitle>
                            <DialogDescription>
                              Fill in the details below to schedule your appointment
                            </DialogDescription>
                          </DialogHeader>
                          <div className="space-y-4 py-4">
                            <div>
                              <label className="text-sm font-medium mb-2 block">Select Date</label>
                              <Select value={selectedDate} onValueChange={setSelectedDate}>
                                <SelectTrigger>
                                  <SelectValue placeholder="Choose a date" />
                                </SelectTrigger>
                                <SelectContent>
                                  {availableDates.map(date => (
                                    <SelectItem key={date} value={date}>
                                      {new Date(date).toLocaleDateString('en-US', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' })}
                                    </SelectItem>
                                  ))}
                                </SelectContent>
                              </Select>
                            </div>

                            <div>
                              <label className="text-sm font-medium mb-2 block">Select Time Slot</label>
                              <div className="grid grid-cols-4 gap-2">
                                {timeSlots.map(time => (
                                  <Button
                                    key={time}
                                    variant={selectedTime === time ? "default" : "outline"}
                                    className="w-full"
                                    onClick={() => setSelectedTime(time)}
                                  >
                                    {time}
                                  </Button>
                                ))}
                              </div>
                            </div>

                            <div>
                              <label className="text-sm font-medium mb-2 block">Describe Your Concern</label>
                              <Textarea
                                placeholder="Tell us what you'd like to discuss... (irrigation issues, crop disease, fertilization plans, etc.)"
                                rows={4}
                                value={concern}
                                onChange={(e) => setConcern(e.target.value)}
                              />
                            </div>

                            <div className="bg-blue-50 p-4 rounded-lg space-y-2">
                              <div className="flex items-center gap-2 text-sm">
                                <Video className="h-4 w-4 text-blue-600" />
                                <span className="font-medium">Consultation will be via Video Call</span>
                              </div>
                              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                                <Mail className="h-4 w-4" />
                                <span>Meeting link will be emailed to you</span>
                              </div>
                              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                                <Phone className="h-4 w-4" />
                                <span>Duration: 30-45 minutes</span>
                              </div>
                            </div>

                            {bookingConfirmed && (
                              <div className="bg-green-50 p-4 rounded-lg flex items-center gap-2">
                                <CheckCircle2 className="h-5 w-5 text-green-600" />
                                <span className="font-medium text-green-800">Appointment booked successfully!</span>
                              </div>
                            )}
                          </div>
                          <DialogFooter>
                            <Button variant="outline" onClick={() => setIsBookingOpen(false)}>Cancel</Button>
                            <Button onClick={handleBooking}>Confirm Booking</Button>
                          </DialogFooter>
                        </DialogContent>
                      </Dialog>
                    </Card>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Right: Consultation Info */}
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Video className="h-5 w-5" /> Why Book a Consultation?
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex items-start gap-3">
                    <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center flex-shrink-0">
                      <CheckCircle2 className="h-5 w-5 text-primary" />
                    </div>
                    <div>
                      <h4 className="font-medium mb-1">Personalized Guidance</h4>
                      <p className="text-sm text-muted-foreground">
                        Get expert advice tailored to your specific farm conditions, crop types, and location.
                      </p>
                    </div>
                  </div>

                  <div className="flex items-start gap-3">
                    <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center flex-shrink-0">
                      <CheckCircle2 className="h-5 w-5 text-primary" />
                    </div>
                    <div>
                      <h4 className="font-medium mb-1">Problem Solving</h4>
                      <p className="text-sm text-muted-foreground">
                        Troubleshoot issues like crop diseases, soil problems, irrigation challenges, and pest management.
                      </p>
                    </div>
                  </div>

                  <div className="flex items-start gap-3">
                    <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center flex-shrink-0">
                      <CheckCircle2 className="h-5 w-5 text-primary" />
                    </div>
                    <div>
                      <h4 className="font-medium mb-1">Farm Planning</h4>
                      <p className="text-sm text-muted-foreground">
                        Plan crop rotation, seasonal schedules, budget allocation, and resource management strategies.
                      </p>
                    </div>
                  </div>

                  <div className="flex items-start gap-3">
                    <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center flex-shrink-0">
                      <CheckCircle2 className="h-5 w-5 text-primary" />
                    </div>
                    <div>
                      <h4 className="font-medium mb-1">Modern Techniques</h4>
                      <p className="text-sm text-muted-foreground">
                        Learn about latest farming technologies, efficient irrigation methods, and organic practices.
                      </p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Calendar className="h-5 w-5" /> Consultation Process
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="flex items-center gap-3">
                    <div className="w-6 h-6 rounded-full bg-primary text-white flex items-center justify-center text-sm font-medium">1</div>
                    <span className="text-sm">Fill your farm details above</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="w-6 h-6 rounded-full bg-primary text-white flex items-center justify-center text-sm font-medium">2</div>
                    <span className="text-sm">Choose an expert specialist</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="w-6 h-6 rounded-full bg-primary text-white flex items-center justify-center text-sm font-medium">3</div>
                    <span className="text-sm">Select date and time slot</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="w-6 h-6 rounded-full bg-primary text-white flex items-center justify-center text-sm font-medium">4</div>
                    <span className="text-sm">Describe your concern</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="w-6 h-6 rounded-full bg-primary text-white flex items-center justify-center text-sm font-medium">5</div>
                    <span className="text-sm">Receive meeting link via email</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="w-6 h-6 rounded-full bg-primary text-white flex items-center justify-center text-sm font-medium">6</div>
                    <span className="text-sm">Attend video consultation</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PlanningConsultation;
