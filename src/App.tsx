import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Navigation from "@/components/Navigation";
import Index from "./pages/Index";
import Weather from "./pages/Weather";
import MarketPrices from "./pages/MarketPrices";
import CropCalendar from "./pages/CropCalendar";
import PestControl from "./pages/PestControl";
import Equipment from "./pages/Equipment";
import PlanningConsultation from "./pages/PlanningConsultation";
import KnowledgeBase from "./pages/KnowledgeBase";
import NotFound from "./pages/NotFound";

const queryClient = new QueryClient();

// Layout component to wrap all pages
const Layout = ({ children }: { children: React.ReactNode }) => (
  <div className="min-h-screen flex flex-col">
    {/* Top Navigation */}
    <div className="sticky top-0 z-40 bg-white/80 backdrop-blur border-b">
      <div className="w-full px-4">
        <Navigation />
      </div>
    </div>

    {/* Main Content */}
    <main className="flex-1 w-full px-4 py-6">
      {children}
    </main>

    {/* Footer */}
    <footer className="bg-white border-t p-4">
      <div className="w-full px-4 text-center text-sm text-gray-500">
        © {new Date().getFullYear()} AgroMind Grow. All rights reserved.
      </div>
    </footer>
  </div>
);

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <Layout>
          <Routes>
            <Route path="/" element={<Index />} />
            <Route path="/weather" element={<Weather />} />
            <Route path="/market-prices" element={<MarketPrices />} />
            <Route path="/crop-calendar" element={<CropCalendar />} />
            <Route path="/pest-control" element={<PestControl />} />
            <Route path="/equipment" element={<Equipment />} />
            <Route path="/planning" element={<PlanningConsultation />} />
            <Route path="/knowledge-base" element={<KnowledgeBase />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </Layout>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;