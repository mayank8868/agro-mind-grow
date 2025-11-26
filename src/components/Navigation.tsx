import { Link, useLocation } from "react-router-dom";
import {
  Cloud,
  TrendingUp,
  Calendar,
  Bug,
  Wrench,
  Users,
  BookOpen,
  Leaf
} from "lucide-react";

const Navigation = () => {
  const location = useLocation();

  const navItems = [
    { to: "/", icon: Leaf, label: "Dashboard" },
    { to: "/pest-control", icon: Bug, label: "Disease Prediction" },
    { to: "/weather", icon: Cloud, label: "Weather" },
    { to: "/market-prices", icon: TrendingUp, label: "Market Prices" },
    { to: "/crop-calendar", icon: Calendar, label: "Crop Calendar" },
    { to: "/equipment", icon: Wrench, label: "Equipment" },
    { to: "/planning", icon: Users, label: "Planning & Consultation" },
    { to: "/knowledge-base", icon: BookOpen, label: "Knowledge Base" },
  ];

  return (
    <nav className="w-full">
      <div className="h-14 flex items-center justify-between">
        <Link to="/" className="flex items-center gap-2 text-foreground">
          <Leaf className="h-5 w-5 text-primary" />
          <span className="text-base font-semibold">AgroMind Grow</span>
        </Link>
        <div className="hidden md:flex items-center gap-6">
          {navItems.map((item) => {
            const active = location.pathname === item.to;
            return (
              <Link
                key={item.to}
                to={item.to}
                className={
                  `text-sm transition-colors hover:text-primary ${active ? "text-primary font-medium" : "text-muted-foreground"}`
                }
              >
                {item.label}
              </Link>
            );
          })}
        </div>
      </div>
    </nav>
  );
};

export default Navigation;