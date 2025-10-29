import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Wrench, Star, Filter, Search, ShoppingCart, Heart } from "lucide-react";import { useEffect, useMemo, useState } from "react";

const Equipment = () => {
  const equipmentCategories = [
    { name: "Tractors", count: 245, icon: "🚜" },
    { name: "Harvesters", count: 89, icon: "🌾" },
    { name: "Irrigation", count: 156, icon: "💧" },
    { name: "Tillage", count: 203, icon: "🪓" },
    { name: "Seeding", count: 134, icon: "🌱" },
    { name: "Hand Tools", count: 78, icon: "🔧" }
  ];

  type Equip = { id:number; name:string; brand:string; category:string; priceINR:number; rating:number; reviews:number; features:string[]; image:string; availability:string; financing:string };
  const [all, setAll] = useState<Equip[]>([]);
  const [query, setQuery] = useState("");
  const [cat, setCat] = useState("all-categories");
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    fetch('/equipment.json')
      .then(r => r.json())
      .then((data: Equip[]) => setAll(data))
      .catch(() => setAll([]))
      .finally(() => setLoading(false));
  }, []);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    const listByCat = cat === 'all-categories' ? all : all.filter(e => e.category.toLowerCase() === cat.toLowerCase());
    return listByCat.filter(e => !q || `${e.name} ${e.brand} ${e.category}`.toLowerCase().includes(q));
  }, [all, query, cat]);

  const serviceProviders = [
    {
      name: "AgriRent Solutions",
      location: "Punjab, Haryana",
      rating: 4.7,
      services: ["Tractor Rental", "Harvester Rental", "Custom Farming"],
      priceRange: "₹800-2000/hour",
      contact: "+91 98765 43210"
    },
    {
      name: "Farm Equipment Hub",
      location: "UP, Bihar",
      rating: 4.4,
      services: ["Equipment Sales", "Repair & Maintenance", "Spare Parts"],
      priceRange: "₹500-1500/hour",
      contact: "+91 87654 32109"
    }
  ];

  return (
    <div className="min-h-screen bg-background">
      
      <div className="w-full px-4 py-8">
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-2">Equipment Catalog</h1>
          <p className="text-muted-foreground text-lg">
            Find, compare, and purchase modern farming equipment
          </p>
        </div>

        {/* Search and Filters */}
        <Card className="mb-8">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Search className="h-5 w-5" />
              Search Equipment
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <Input placeholder="Search equipment..." className="md:col-span-2" value={query} onChange={(e) => setQuery(e.target.value)} />
              <Select value={cat} onValueChange={setCat}>
                <SelectTrigger>
                  <SelectValue placeholder="All Categories" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all-categories">All Categories</SelectItem>
                  <SelectItem value="Tractors">Tractors</SelectItem>
                  <SelectItem value="Harvesters">Harvesters</SelectItem>
                  <SelectItem value="Irrigation">Irrigation</SelectItem>
                  <SelectItem value="Tillage">Tillage</SelectItem>
                  <SelectItem value="Seeding">Seeding</SelectItem>
                  <SelectItem value="Hand Tools">Hand Tools</SelectItem>
                </SelectContent>
              </Select>
              <Button>
                <Filter className="mr-2 h-4 w-4" />
                Filter
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Equipment Categories */}
        <Card className="mb-8">
          <CardHeader>
            <CardTitle>Equipment Categories</CardTitle>
            <CardDescription>Browse by equipment type</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
              {equipmentCategories.map((category, index) => (
                <div key={index} className="text-center p-4 border rounded-lg hover:bg-muted cursor-pointer transition-colors">
                  <div className="text-3xl mb-2">{category.icon}</div>
                  <h3 className="font-semibold">{category.name}</h3>
                  <p className="text-sm text-muted-foreground">{category.count} items</p>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Featured Equipment (live from JSON) */}
        <Card className="mb-8">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Wrench className="h-5 w-5" />
              Featured Equipment
            </CardTitle>
            <CardDescription>Top-rated farming equipment and machinery</CardDescription>
          </CardHeader>
          <CardContent>
            {loading && <div className="text-sm text-muted-foreground">Loading equipment...</div>}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {filtered.map((item) => (
                <div key={item.id} className="border rounded-lg p-4">
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex items-center gap-3">
                      <span className="text-3xl">{item.image}</span>
                      <div>
                        <h3 className="font-semibold text-lg">{item.name}</h3>
                        <Badge variant="outline">{item.category}</Badge>
                      </div>
                    </div>
                    <Button variant="ghost" size="sm">
                      <Heart className="h-4 w-4" />
                    </Button>
                  </div>

                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-2xl font-bold text-primary">₹{item.priceINR.toLocaleString('en-IN')}</span>
                      <div className="flex items-center gap-1">
                        <Star className="h-4 w-4 fill-yellow-400 text-yellow-400" />
                        <span className="font-semibold">{item.rating}</span>
                        <span className="text-sm text-muted-foreground">({item.reviews})</span>
                      </div>
                    </div>

                    <div>
                      <h4 className="font-medium mb-2">Key Features:</h4>
                      <div className="flex flex-wrap gap-1">
                        {item.features.map((feature, idx) => (
                          <Badge key={idx} variant="secondary" className="text-xs">
                            {feature}
                          </Badge>
                        ))}
                      </div>
                    </div>

                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-4">
                        <Badge variant={item.availability === "In Stock" ? "default" : "secondary"}>
                          {item.availability}
                        </Badge>
                        <span className="text-sm text-green-600">{item.financing}</span>
                      </div>
                    </div>

                    <div className="flex gap-2">
                      <Button className="flex-1">
                        <ShoppingCart className="mr-2 h-4 w-4" />
                        Buy Now
                      </Button>
                      <Button variant="outline" className="flex-1">
                        View Details
                      </Button>
                    </div>
                  </div>
                </div>
              ))}
              {filtered.length === 0 && (
                <div className="text-sm text-muted-foreground">No equipment found for the selected filters.</div>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Service Providers */}
        <Card className="mb-8">
          <CardHeader>
            <CardTitle>Equipment Rental & Services</CardTitle>
            <CardDescription>Find equipment rental and maintenance services near you</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {serviceProviders.map((provider, index) => (
                <div key={index} className="border rounded-lg p-4">
                  <div className="flex items-start justify-between mb-3">
                    <div>
                      <h3 className="font-semibold text-lg">{provider.name}</h3>
                      <p className="text-sm text-muted-foreground">{provider.location}</p>
                    </div>
                    <div className="text-right">
                      <div className="flex items-center gap-1">
                        <Star className="h-4 w-4 fill-yellow-400 text-yellow-400" />
                        <span className="font-semibold">{provider.rating}</span>
                      </div>
                      <p className="text-sm text-muted-foreground">{provider.priceRange}</p>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <h4 className="font-medium mb-2">Services Offered:</h4>
                      <div className="flex flex-wrap gap-1">
                        {provider.services.map((service, idx) => (
                          <Badge key={idx} variant="outline">
                            {service}
                          </Badge>
                        ))}
                      </div>
                    </div>
                    <div className="flex items-center justify-between">
                      <div>
                        <h4 className="font-medium">Contact:</h4>
                        <p className="text-sm text-muted-foreground">{provider.contact}</p>
                      </div>
                      <Button>Get Quote</Button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Financing Options */}
        <Card>
          <CardHeader>
            <CardTitle>Financing & Government Schemes</CardTitle>
            <CardDescription>Available financing options for equipment purchase</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="p-4 bg-green-50 rounded-lg">
                <h4 className="font-semibold text-green-800 mb-2">NABARD Subsidy</h4>
                <p className="text-sm text-green-700 mb-3">Up to 50% subsidy on agricultural equipment</p>
                <Button variant="outline" size="sm">Learn More</Button>
              </div>
              <div className="p-4 bg-blue-50 rounded-lg">
                <h4 className="font-semibold text-blue-800 mb-2">Bank Loans</h4>
                <p className="text-sm text-blue-700 mb-3">Easy EMI options with low interest rates</p>
                <Button variant="outline" size="sm">Apply Now</Button>
              </div>
              <div className="p-4 bg-purple-50 rounded-lg">
                <h4 className="font-semibold text-purple-800 mb-2">Equipment Leasing</h4>
                <p className="text-sm text-purple-700 mb-3">Flexible leasing options for expensive machinery</p>
                <Button variant="outline" size="sm">Get Quote</Button>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default Equipment;