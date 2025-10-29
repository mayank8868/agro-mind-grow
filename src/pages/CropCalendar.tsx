import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Calendar as CalIcon, Plus, ChevronLeft, ChevronRight, Check, Trash2 } from "lucide-react";
import React, { useMemo, useState } from "react";
import { startOfMonth, endOfMonth, startOfWeek, endOfWeek, addDays, addMonths, format, isSameMonth, isToday } from "date-fns";
 

const CropCalendar = () => {
  // Simplified: no filters/overview/weather blocks

  // --- Crop Scheduler (monthly) ---
  type TaskType = 'Irrigation' | 'Fertilization' | 'Weeding' | 'Pest control' | 'Sowing' | 'Harvest' | 'Other';
  type CalendarTask = {
    id: string;
    crop: string;             // which crop this task belongs to
    dateISO: string;          // yyyy-MM-dd
    type: TaskType;
    title: string;
    notes?: string;
    done: boolean;
  };

  function storageKey(cropName: string | undefined) {
    return `crop-schedule:v1:${cropName ?? 'all'}`;
  }

  function loadTasks(cropName: string | undefined): CalendarTask[] {
    try {
      const raw = localStorage.getItem(storageKey(cropName));
      return raw ? (JSON.parse(raw) as CalendarTask[]) : [];
    } catch {
      return [];
    }
  }

  function saveTasks(cropName: string | undefined, tasks: CalendarTask[]) {
    try { localStorage.setItem(storageKey(cropName), JSON.stringify(tasks)); } catch {}
  }

  const [currentMonthDate, setCurrentMonthDate] = useState<Date>(startOfMonth(new Date()));
  const [tasksState, setTasksState] = useState<CalendarTask[]>(() => []);

  // Use a generic crop context for scheduling (no filters required)
  const activeCropForSchedule = 'General';
  // Reload tasks when crop changes
  // eslint-disable-next-line react-hooks/exhaustive-deps
  React.useEffect(() => { setTasksState(loadTasks(activeCropForSchedule)); }, [activeCropForSchedule]);
  const cropTasks: CalendarTask[] = useMemo(() => {
    return tasksState.filter(t => !activeCropForSchedule || t.crop === activeCropForSchedule);
  }, [tasksState, activeCropForSchedule]);

  const monthMatrix: Date[][] = useMemo(() => {
    const start = startOfWeek(startOfMonth(currentMonthDate), { weekStartsOn: 1 });
    const end = endOfWeek(endOfMonth(currentMonthDate), { weekStartsOn: 1 });
    const days: Date[][] = [];
    let cur = start;
    while (cur <= end) {
      const week: Date[] = [];
      for (let i = 0; i < 7; i++) { week.push(cur); cur = addDays(cur, 1); }
      days.push(week);
    }
    return days;
  }, [currentMonthDate]);

  // Dialog state
  const [isOpen, setIsOpen] = useState<boolean>(false);
  const [draftDate, setDraftDate] = useState<string>(format(new Date(), 'yyyy-MM-dd'));
  const [draftType, setDraftType] = useState<TaskType>('Irrigation');
  const [draftTitle, setDraftTitle] = useState<string>('');
  const [draftNotes, setDraftNotes] = useState<string>('');
  const [editId, setEditId] = useState<string | null>(null);

  function openNewTask(date: Date) {
    setEditId(null);
    setDraftDate(format(date, 'yyyy-MM-dd'));
    setDraftType('Irrigation');
    setDraftTitle('');
    setDraftNotes('');
    setIsOpen(true);
  }

  function openEditTask(task: CalendarTask) {
    setEditId(task.id);
    setDraftDate(task.dateISO);
    setDraftType(task.type);
    setDraftTitle(task.title);
    setDraftNotes(task.notes ?? '');
    setIsOpen(true);
  }

  function upsertTask() {
    if (!activeCropForSchedule) return setIsOpen(false);
    const next: CalendarTask[] = [...tasksState];
    if (editId) {
      const idx = next.findIndex(t => t.id === editId);
      if (idx >= 0) next[idx] = { ...next[idx], crop: activeCropForSchedule, dateISO: draftDate, type: draftType, title: draftTitle.trim() || draftType, notes: draftNotes };
    } else {
      next.push({ id: crypto.randomUUID(), crop: activeCropForSchedule, dateISO: draftDate, type: draftType, title: draftTitle.trim() || draftType, notes: draftNotes, done: false });
    }
    saveTasks(activeCropForSchedule, next);
    setTasksState(next);
    setIsOpen(false);
  }

  function toggleDone(task: CalendarTask) {
    const next = [...tasksState];
    const idx = next.findIndex(t => t.id === task.id);
    if (idx >= 0) next[idx] = { ...next[idx], done: !next[idx].done };
    saveTasks(activeCropForSchedule, next);
    setTasksState(next);
  }

  function deleteTask(task: CalendarTask) {
    const next = tasksState.filter(t => t.id !== task.id);
    saveTasks(activeCropForSchedule, next);
    setTasksState(next);
  }

  return (
    <div className="min-h-screen bg-background">
      
      <div className="w-full px-4 py-8">
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-2">Crop Calendar & Planning</h1>
          <p className="text-muted-foreground text-lg">
            Plan your farming activities with seasonal crop schedules
          </p>
        </div>

        {/* Simplified page: only the scheduler remains */}

        {/* Crop Scheduler (Monthly) */}
        <Card className="mb-8">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div>
            <CardTitle className="flex items-center gap-2">
              <CalIcon className="h-5 w-5" />
                  Monthly Crop Scheduler
            </CardTitle>
                <CardDescription>Plan irrigation, fertilization, harvest and more by day</CardDescription>
              </div>
              <div className="flex items-center gap-2">
                <Button variant="outline" size="icon" onClick={() => setCurrentMonthDate(addMonths(currentMonthDate, -1))}><ChevronLeft className="h-4 w-4" /></Button>
                <div className="text-sm font-medium w-36 text-center">{format(currentMonthDate, 'MMMM yyyy')}</div>
                <Button variant="outline" size="icon" onClick={() => setCurrentMonthDate(addMonths(currentMonthDate, 1))}><ChevronRight className="h-4 w-4" /></Button>
                <Button onClick={() => openNewTask(new Date())} className="ml-2"><Plus className="h-4 w-4 mr-2" />Add Task</Button>
              </div>
            </div>
          </CardHeader>
          <CardContent>
                      <div>
                {/* Weekday headers */}
                <div className="grid grid-cols-7 text-xs text-muted-foreground mb-2">
                  {['Mon','Tue','Wed','Thu','Fri','Sat','Sun'].map(d => (<div key={d} className="text-center">{d}</div>))}
                </div>
                {/* Calendar grid */}
                <div className="grid grid-cols-7 gap-1">
                  {monthMatrix.flat().map((day, idx) => {
                    const dateISO = format(day, 'yyyy-MM-dd');
                    const ds = cropTasks.filter(t => t.dateISO === dateISO);
                    const muted = !isSameMonth(day, currentMonthDate);
                    return (
                      <div key={idx} className={`min-h-[110px] p-2 border rounded hover:bg-accent/30 cursor-pointer ${muted ? 'opacity-50' : ''}`} onClick={() => openNewTask(day)}>
                        <div className={`text-xs mb-1 ${isToday(day) ? 'font-semibold text-primary' : ''}`}>{format(day, 'd')}</div>
                        <div className="space-y-1">
                          {ds.slice(0,3).map(task => (
                            <div key={task.id} className={`px-2 py-1 rounded text-[11px] flex items-center justify-between ${task.done ? 'bg-green-100 text-green-800' : 'bg-muted text-foreground'}`} onClick={(e) => { e.stopPropagation(); openEditTask(task); }}>
                              <span className="truncate mr-2">{task.title}</span>
                              {task.done && <Check className="h-3 w-3" />}
                            </div>
                          ))}
                          {ds.length > 3 && (
                            <div className="text-[11px] text-muted-foreground">+{ds.length - 3} more</div>
                          )}
                      </div>
                      </div>
                    );
                  })}
                    </div>
                </div>
          </CardContent>
        </Card>

        {/* Add/Edit Task Dialog */}
        <Dialog open={isOpen} onOpenChange={setIsOpen}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>{editId ? 'Edit Task' : 'Add Task'}</DialogTitle>
              <DialogDescription>Schedule crop operations on specific dates.</DialogDescription>
            </DialogHeader>
            <div className="grid gap-3 py-2">
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="text-xs text-muted-foreground">Date</label>
                  <Input type="date" value={draftDate} onChange={(e) => setDraftDate(e.target.value)} />
                </div>
                <div>
                  <label className="text-xs text-muted-foreground">Type</label>
                  <Select value={draftType} onValueChange={(v) => setDraftType(v as TaskType)}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {['Irrigation','Fertilization','Weeding','Pest control','Sowing','Harvest','Other'].map((t) => (
                        <SelectItem key={t} value={t}>{t}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>
              <div>
                <label className="text-xs text-muted-foreground">Title</label>
                <Input placeholder="e.g., Urea application (45kg/acre)" value={draftTitle} onChange={(e) => setDraftTitle(e.target.value)} />
              </div>
              <div>
                <label className="text-xs text-muted-foreground">Notes</label>
                <Textarea placeholder="Add any instructions or quantities" value={draftNotes} onChange={(e) => setDraftNotes(e.target.value)} />
              </div>
            </div>
            <DialogFooter>
              {editId && (
                <Button variant="destructive" onClick={() => { const t = loadTasks(activeCropForSchedule).find(x=>x.id===editId); if (t) { deleteTask(t); setIsOpen(false);} }}>
                  <Trash2 className="h-4 w-4 mr-2" /> Delete
                </Button>
              )}
              <Button onClick={upsertTask}>{editId ? 'Save' : 'Add Task'}</Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        {/* Weather/advisory blocks removed */}
      </div>
    </div>
  );
};

export default CropCalendar;