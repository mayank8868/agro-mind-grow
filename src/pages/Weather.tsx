import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Cloud, Sun, CloudRain, CloudLightning, Droplets, Wind, Eye, Search } from "lucide-react";
import { useEffect, useState } from "react";
import { fetchWeatherByCity, codeToCondition, WeatherResult } from "@/lib/weather";

function countryCodeFromValue(v: string): string | undefined {
  switch (v) {
    case "india":
      return "IN";
    case "usa":
      return "US";
    case "uk":
      return "GB";
    default:
      return undefined;
  }
}

const Weather = () => {
  const [cityInput, setCityInput] = useState("Delhi");
  const [country, setCountry] = useState("india");
  const [queryCity, setQueryCity] = useState("Delhi");
  const [data, setData] = useState<WeatherResult | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [isFetching, setIsFetching] = useState<boolean>(false);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    let cancelled = false;
    setIsLoading(true);
    setError(null);
    fetchWeatherByCity(queryCity, countryCodeFromValue(country))
      .then((res) => {
        if (cancelled) return;
        setData(res);
      })
      .catch((e: any) => {
        if (cancelled) return;
        setError(e instanceof Error ? e : new Error(String(e)));
      })
      .finally(() => {
        if (cancelled) return;
        setIsLoading(false);
      });
    return () => { cancelled = true; };
  }, [queryCity, country]);

  const current = data?.current;
  const forecast = data?.forecast ?? [];

  function handleSearch() {
    if (!cityInput.trim()) return;
    setIsFetching(true);
    setQueryCity(cityInput.trim());
    // isFetching will be reset once effect finishes
    setTimeout(() => setIsFetching(false), 500);
  }

  function codeToIcon(code: number) {
    if (code === 0) return Sun;
    if ([1, 2, 3].includes(code)) return Cloud;
    if ([45, 48].includes(code)) return Cloud;
    if ([51, 53, 55, 61, 63, 65, 80, 81, 82].includes(code)) return CloudRain;
    if ([95, 96, 99].includes(code)) return CloudLightning;
    return Cloud;
  }

  return (
    <div className="min-h-screen bg-background">
      
      <div className="w-full px-4 py-8">
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-2">Weather & Climate Analysis</h1>
          <p className="text-muted-foreground text-lg">
            Real-time weather data and forecasts for better crop planning
          </p>
        </div>

        {/* Location Search */}
        <Card className="mb-8">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Search className="h-5 w-5" />
              Location Search
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex gap-4">
              <Input
                placeholder="Enter city"
                className="flex-1"
                value={cityInput}
                onChange={(e) => setCityInput(e.target.value)}
              />
              <Select value={country} onValueChange={setCountry}>
                <SelectTrigger className="w-48">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="india">India</SelectItem>
                  <SelectItem value="usa">United States</SelectItem>
                  <SelectItem value="uk">United Kingdom</SelectItem>
                </SelectContent>
              </Select>
              <Button onClick={handleSearch} disabled={isFetching}>Search</Button>
            </div>
            <div className="text-sm text-muted-foreground mt-2">
              Try examples: Delhi, Mumbai, Kolkata, Chennai, Bengaluru
            </div>
          </CardContent>
        </Card>

        {/* Current Weather */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
          <Card className="lg:col-span-2">
            <CardHeader>
              <CardTitle>Current Weather</CardTitle>
              <CardDescription>
                {isLoading ? "Loading..." : (error ? "" : current?.locationLabel)}
              </CardDescription>
            </CardHeader>
            <CardContent>
              {error && (
                <div className="text-destructive">{(error as Error)?.message || "Failed to load weather"}</div>
              )}
              {!error && (
                <div>
                  <div className="flex items-center justify-between mb-6">
                    <div>
                      <div className="text-6xl font-bold">{isLoading ? "--" : `${current?.temperatureC ?? "--"}°C`}</div>
                      <div className="text-lg text-muted-foreground">
                        {isLoading ? "" : codeToCondition(current?.conditionCode ?? 0)}
                      </div>
                    </div>
                    <Cloud className="h-24 w-24 text-primary" />
                  </div>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="flex items-center gap-2">
                      <Droplets className="h-5 w-5 text-blue-500" />
                      <div>
                        <div className="text-sm text-muted-foreground">Humidity</div>
                        <div className="font-semibold">{isLoading ? "--" : `${current?.humidityPercent ?? "--"}%`}</div>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <Wind className="h-5 w-5 text-green-500" />
                      <div>
                        <div className="text-sm text-muted-foreground">Wind Speed</div>
                        <div className="font-semibold">{isLoading ? "--" : `${current?.windSpeedKmh ?? "--"} km/h`}</div>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <Eye className="h-5 w-5 text-purple-500" />
                      <div>
                        <div className="text-sm text-muted-foreground">Visibility</div>
                        <div className="font-semibold">{current?.visibilityKm ?? "--"} km</div>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <Sun className="h-5 w-5 text-orange-500" />
                      <div>
                        <div className="text-sm text-muted-foreground">UV Index</div>
                        <div className="font-semibold">{current?.uvIndex ?? "--"}</div>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Farming Recommendations</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="p-3 bg-green-100 rounded-lg">
                <h4 className="font-semibold text-green-800">Ideal for Irrigation</h4>
                <p className="text-sm text-green-700">Low humidity levels suggest good irrigation conditions</p>
              </div>
              <div className="p-3 bg-yellow-100 rounded-lg">
                <h4 className="font-semibold text-yellow-800">Moderate UV Exposure</h4>
                <p className="text-sm text-yellow-700">Good conditions for most crops, consider shade for sensitive plants</p>
              </div>
              <div className="p-3 bg-blue-100 rounded-lg">
                <h4 className="font-semibold text-blue-800">Wind Conditions</h4>
                <p className="text-sm text-blue-700">Favorable wind speed for natural ventilation</p>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* 5-Day Forecast */}
        <Card className="mb-8">
          <CardHeader>
            <CardTitle>5-Day Forecast</CardTitle>
            <CardDescription>Plan your farming activities ahead</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
              {forecast.map((day, index) => {
                const Icon = codeToIcon(day.conditionCode);
                return (
                  <div key={index} className="text-center p-4 rounded-lg bg-muted">
                    <div className="font-semibold mb-2">{day.dayLabel}</div>
                    <Icon className="h-8 w-8 mx-auto mb-2 text-primary" />
                    <div className="text-sm text-muted-foreground mb-1">{codeToCondition(day.conditionCode)}</div>
                    <div className="font-semibold">
                      {day.highC}°C / {day.lowC}°C
                    </div>
                  </div>
                );
              })}
            </div>
          </CardContent>
        </Card>

        {/* Agricultural Alerts */}
        <Card>
          <CardHeader>
            <CardTitle>Agricultural Weather Alerts</CardTitle>
            <CardDescription>Important weather notifications for farmers</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="p-4 border-l-4 border-green-500 bg-green-50">
                <h4 className="font-semibold text-green-800">Favorable Planting Conditions</h4>
                <p className="text-green-700">Next 3 days show ideal temperature and humidity for rice plantation</p>
                <p className="text-sm text-green-600 mt-1">Valid until: March 25, 2024</p>
              </div>
              <div className="p-4 border-l-4 border-orange-500 bg-orange-50">
                <h4 className="font-semibold text-orange-800">Heat Wave Warning</h4>
                <p className="text-orange-700">Expected high temperatures next week. Increase irrigation frequency</p>
                <p className="text-sm text-orange-600 mt-1">Valid from: March 26-30, 2024</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default Weather;