import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { MapPin, RefreshCcw, Search, TrendingUp, TrendingDown, Minus, BarChart3, Calendar } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { fetchMarketPrices, MarketPriceRecord } from "@/lib/market";

const MarketPrices = () => {
  const [query, setQuery] = useState<string>("");
  const [stateFilter, setStateFilter] = useState<string>("All");
  const [page, setPage] = useState<number>(0);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [records, setRecords] = useState<MarketPriceRecord[]>([]);

  useEffect(() => {
    load();
  }, []);

  async function load() {
    setLoading(true);
    setError(null);
    try {
      const base = { limit: 500, state: stateFilter === 'All' ? undefined : stateFilter } as any;
      const [p1, p2] = await Promise.all([
        fetchMarketPrices({ ...base, offset: page * 500 }),
        fetchMarketPrices({ ...base, offset: page * 500 + 500 }),
      ]);
      let all: MarketPriceRecord[] = [...(p1.records || []), ...(p2.records || [])];
      if (stateFilter !== 'All') {
        all = all.filter((r) => (r.state || '').toLowerCase() === stateFilter.toLowerCase());
      }
      all = all.sort((a, b) => (a.arrival_date < b.arrival_date ? 1 : a.arrival_date > b.arrival_date ? -1 : (b.modal_price ?? 0) - (a.modal_price ?? 0)));
      setRecords(all);
    } catch (e: any) {
      setError(e?.message || "Failed to load market data");
      setRecords([]);
    } finally {
      setLoading(false);
    }
  }

  const filtered = useMemo(() => {
    const normalize = (s?: string) => (s || "")
      .toLowerCase()
      .trim()
      .replace(/\s+/g, "")
      .replace(/[^a-z0-9]/g, "");
    const q = normalize(query);
    if (!q) return records;
    return records.filter((r) => normalize([r.commodity, r.variety, r.market, r.district, r.state].join(" ")).includes(q));
  }, [records, query]);

  function formatINR(value?: number): string {
    if (value == null || !Number.isFinite(value)) return "--";
    return value.toLocaleString("en-IN");
  }

  function parseDateFlex(d: string): number {
    // Returns ms since epoch; supports YYYY-MM-DD and DD-MM-YYYY
    const iso = Date.parse(d);
    if (!Number.isNaN(iso)) return iso;
    const m = d.match(/^(\d{2})[-\/](\d{2})[-\/](\d{4})$/);
    if (m) {
      const [, dd, mm, yyyy] = m;
      return Date.parse(`${yyyy}-${mm}-${dd}`);
    }
    return 0;
  }

  function priceOf(r: MarketPriceRecord): number | undefined {
    return r.modal_price ?? r.max_price ?? r.min_price ?? undefined;
  }

  const stats = useMemo(() => {
    let up = 0, down = 0, flat = 0;
    let sum = 0, count = 0;
    let latestMs = 0;

    const groups = new Map<string, MarketPriceRecord[]>();
    for (const r of filtered) {
      const p = priceOf(r);
      if (typeof p === 'number') { sum += p; count++; }
      const ms = parseDateFlex(r.arrival_date);
      if (ms > latestMs) latestMs = ms;
      const key = `${r.state}|${r.market}|${r.commodity}|${r.variety ?? ''}`;
      const arr = groups.get(key) ?? [];
      arr.push(r);
      groups.set(key, arr);
    }

    for (const arr of groups.values()) {
      if (arr.length < 2) continue;
      arr.sort((a, b) => parseDateFlex(b.arrival_date) - parseDateFlex(a.arrival_date));
      const p1 = priceOf(arr[0]);
      const p2 = priceOf(arr[1]);
      if (typeof p1 !== 'number' || typeof p2 !== 'number') continue;
      if (p1 > p2) up++; else if (p1 < p2) down++; else flat++;
    }

    return {
      up, down, flat,
      avg: count ? Math.round(sum / count) : undefined,
      latestDate: latestMs ? new Date(latestMs).toLocaleDateString() : undefined,
    };
  }, [filtered]);

  return (
    <div className="min-h-screen bg-background">

      <div className="w-full px-4 py-8">
        <div className="mb-6 grid grid-cols-1 md:grid-cols-3 gap-3 items-center">
          <h1 className="text-3xl font-bold">Market Prices</h1>
          <div>
            <Select value={stateFilter} onValueChange={setStateFilter}>
              <SelectTrigger>
                <SelectValue placeholder="All States" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="All">All States</SelectItem>
                <SelectItem value="Punjab">Punjab</SelectItem>
                <SelectItem value="Uttar Pradesh">Uttar Pradesh</SelectItem>
                <SelectItem value="Haryana">Haryana</SelectItem>
                <SelectItem value="Gujarat">Gujarat</SelectItem>
                <SelectItem value="Madhya Pradesh">Madhya Pradesh</SelectItem>
                <SelectItem value="Rajasthan">Rajasthan</SelectItem>
                <SelectItem value="Maharashtra">Maharashtra</SelectItem>
                <SelectItem value="Andhra Pradesh">Andhra Pradesh</SelectItem>
                <SelectItem value="Assam">Assam</SelectItem>
                <SelectItem value="Bihar">Bihar</SelectItem>
                <SelectItem value="Karnataka">Karnataka</SelectItem>
                <SelectItem value="Tamil Nadu">Tamil Nadu</SelectItem>
                <SelectItem value="Telangana">Telangana</SelectItem>
                <SelectItem value="West Bengal">West Bengal</SelectItem>
                <SelectItem value="Odisha">Odisha</SelectItem>
                <SelectItem value="Chhattisgarh">Chhattisgarh</SelectItem>
                <SelectItem value="Jharkhand">Jharkhand</SelectItem>
                <SelectItem value="Uttarakhand">Uttarakhand</SelectItem>
                <SelectItem value="Himachal Pradesh">Himachal Pradesh</SelectItem>
                <SelectItem value="Jammu and Kashmir">Jammu and Kashmir</SelectItem>
                <SelectItem value="Kerala">Kerala</SelectItem>
                <SelectItem value="Delhi">Delhi</SelectItem>
                <SelectItem value="Chandigarh">Chandigarh</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div className="text-right">
            <div className="flex gap-2 justify-end">
              <Button onClick={() => { setPage((p) => Math.max(0, p - 1)); load(); }} variant="outline" disabled={loading || page === 0}>Prev</Button>
              <Button onClick={() => { setPage((p) => p + 1); load(); }} variant="outline" disabled={loading}>Next</Button>
              <Button onClick={load} variant="outline" disabled={loading}>
                <RefreshCcw className="h-4 w-4 mr-2" /> Refresh
              </Button>
            </div>
          </div>
        </div>

        <Card className="mb-6">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <Search className="h-4 w-4" /> Search by commodity, market, district, or state
            </CardTitle>
          </CardHeader>
          <CardContent>
            <Input
              placeholder="e.g., Wheat, Soyabean, Indore, Punjab"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Latest Market Rates</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-3 mb-4">
              <div className="p-3 border rounded-lg">
                <div className="flex items-center justify-between text-sm text-muted-foreground mb-1">
                  <span>Price Up</span>
                  <TrendingUp className="h-4 w-4 text-green-600" />
                </div>
                <div className="text-2xl font-bold text-green-700">{stats.up}</div>
              </div>
              <div className="p-3 border rounded-lg">
                <div className="flex items-center justify-between text-sm text-muted-foreground mb-1">
                  <span>Price Down</span>
                  <TrendingDown className="h-4 w-4 text-red-600" />
                </div>
                <div className="text-2xl font-bold text-red-700">{stats.down}</div>
              </div>
              <div className="p-3 border rounded-lg">
                <div className="flex items-center justify-between text-sm text-muted-foreground mb-1">
                  <span>Stable</span>
                  <Minus className="h-4 w-4 text-gray-600" />
                </div>
                <div className="text-2xl font-bold">{stats.flat}</div>
              </div>
              <div className="p-3 border rounded-lg">
                <div className="flex items-center justify-between text-sm text-muted-foreground mb-1">
                  <span>Avg. Price</span>
                  <BarChart3 className="h-4 w-4 text-blue-600" />
                </div>
                <div className="text-2xl font-bold">₹{stats.avg ? stats.avg.toLocaleString('en-IN') : '--'}</div>
                <div className="text-xs text-muted-foreground mt-1 flex items-center gap-1">
                  <Calendar className="h-3 w-3" />
                  {stats.latestDate ? `Latest: ${stats.latestDate}` : 'No date'}
                </div>
              </div>
            </div>
            {error && (
              <div className="p-3 rounded border border-destructive text-destructive mb-4 text-sm">{error}</div>
            )}
            {loading && (
              <div className="text-sm text-muted-foreground mb-4">Loading market data...</div>
            )}
            {!loading && !error && filtered.length === 0 && (
              <div className="text-sm text-muted-foreground">No records found. Try a different search.</div>
            )}
            <div className="grid gap-3">
              {filtered.map((r, idx) => (
                <div key={idx} className="grid grid-cols-1 md:grid-cols-12 gap-2 p-3 border rounded-lg">
                  <div className="md:col-span-3 font-semibold">
                    {r.commodity}{r.variety ? ` • ${r.variety}` : ""}
                  </div>
                  <div className="md:col-span-4 text-sm text-muted-foreground flex items-center gap-2">
                    <MapPin className="h-3 w-3" />
                    {r.market}, {r.district}, {r.state}
                  </div>
                  <div className="md:col-span-2 text-sm text-muted-foreground">{r.arrival_date}</div>
                  <div className="md:col-span-3 text-right">
                    <div className="text-xl font-bold">₹{formatINR(r.modal_price ?? r.max_price ?? r.min_price)}</div>
                    <div className="text-xs text-muted-foreground">{r.unit_of_price || "per quintal"}</div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default MarketPrices;