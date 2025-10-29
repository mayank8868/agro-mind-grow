import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { BookOpen, Video, FileText, Users, Search, Clock, ThumbsUp, Eye } from "lucide-react";

const KnowledgeBase = () => {
  const categories = [
    { name: "Crop Management", articles: 145, icon: "🌾", description: "Learn about different crops and their management" },
    { name: "Soil & Fertilizers", articles: 89, icon: "🌱", description: "Soil health and fertilization techniques" },
    { name: "Pest Control", articles: 67, icon: "🐛", description: "Integrated pest management strategies" },
    { name: "Water Management", articles: 54, icon: "💧", description: "Irrigation and water conservation methods" },
    { name: "Equipment & Tools", articles: 78, icon: "🚜", description: "Farming equipment and modern tools" },
    { name: "Market & Finance", articles: 43, icon: "💰", description: "Marketing strategies and financial planning" },
    { name: "Organic Farming", articles: 56, icon: "🌿", description: "Sustainable and organic farming practices" },
    { name: "Weather & Climate", articles: 34, icon: "🌤️", description: "Weather patterns and climate adaptation" }
  ];

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

  const tutorials = [
    {
      title: "Modern Farming Techniques",
      description: "Learn latest farming methods and technologies",
      videos: 12,
      duration: "3 hours",
      difficulty: "Beginner to Advanced",
      instructor: "Agricultural University"
    },
    {
      title: "Organic Farming Certification",
      description: "Complete guide to getting organic certification",
      videos: 8,
      duration: "2 hours",
      difficulty: "Intermediate",
      instructor: "Organic Council of India"
    },
    {
      title: "Farm Business Management",
      description: "Learn to manage your farm as a profitable business",
      videos: 15,
      duration: "4 hours",
      difficulty: "Advanced",
      instructor: "Agricultural Economics Institute"
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

  return (
    <div className="min-h-screen bg-background">
      
      <div className="w-full px-4 py-8">
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-2">Knowledge Base</h1>
          <p className="text-muted-foreground text-lg">
            Comprehensive farming guides, tutorials, and best practices
          </p>
        </div>

        {/* Search */}
        <Card className="mb-8">
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
        <Card className="mb-8">
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
        <Card className="mb-8">
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

        {/* Video Tutorials */}
        <Card className="mb-8">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Video className="h-5 w-5" />
              Video Tutorial Series
            </CardTitle>
            <CardDescription>Comprehensive video courses on farming techniques</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {tutorials.map((tutorial, index) => (
                <div key={index} className="border rounded-lg p-4">
                  <h3 className="font-semibold text-lg mb-2">{tutorial.title}</h3>
                  <p className="text-sm text-muted-foreground mb-4">{tutorial.description}</p>
                  
                  <div className="space-y-2 mb-4">
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">Videos:</span>
                      <span className="font-medium">{tutorial.videos}</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">Duration:</span>
                      <span className="font-medium">{tutorial.duration}</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">Instructor:</span>
                      <span className="font-medium">{tutorial.instructor}</span>
                    </div>
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <Badge className={getDifficultyColor(tutorial.difficulty)}>
                      {tutorial.difficulty}
                    </Badge>
                    <Button size="sm">Start Course</Button>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Community Content */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Users className="h-5 w-5" />
              Community Contributions
            </CardTitle>
            <CardDescription>Knowledge shared by fellow farmers and experts</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="p-4 bg-green-50 rounded-lg">
                <h4 className="font-semibold text-green-800 mb-2">Success Stories</h4>
                <p className="text-sm text-green-700 mb-3">Read how farmers achieved success using modern techniques</p>
                <Button variant="outline" size="sm">Read Stories</Button>
              </div>
              
              <div className="p-4 bg-blue-50 rounded-lg">
                <h4 className="font-semibold text-blue-800 mb-2">Q&A Forum</h4>
                <p className="text-sm text-blue-700 mb-3">Get answers to your farming questions from experts</p>
                <Button variant="outline" size="sm">Visit Forum</Button>
              </div>
              
              <div className="p-4 bg-purple-50 rounded-lg">
                <h4 className="font-semibold text-purple-800 mb-2">Best Practices</h4>
                <p className="text-sm text-purple-700 mb-3">Learn proven techniques from experienced farmers</p>
                <Button variant="outline" size="sm">Explore</Button>
              </div>
              
              <div className="p-4 bg-orange-50 rounded-lg">
                <h4 className="font-semibold text-orange-800 mb-2">Case Studies</h4>
                <p className="text-sm text-orange-700 mb-3">Detailed analysis of farming projects and outcomes</p>
                <Button variant="outline" size="sm">View Cases</Button>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default KnowledgeBase;