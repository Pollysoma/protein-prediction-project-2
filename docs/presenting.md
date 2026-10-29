# Presentation checklist

Distilled from the course material ("How to (not) give a presentation", the RostLab template, and the project-session slides). Start every talk from the RostLab template; the latest deck in [`../presentation/`](../presentation/) uses it.

## Format

- 10-minute talk + 5 minutes of questions. Exam: 15 minutes + 10 minutes of questions per student.
- About one slide per minute.
- Each talk answers: what we worked on (and which part of the project it serves), our biggest success since the last talk, the current roadblock, the next steps and why, and how we evaluated progress.
- Structure: a short recap of the goal → results in a logical order → what they mean for the project → next steps.

## Never go wrong

- [ ] Title slide with title, date, and the name, photo and e-mail of each team member
- [ ] Slide numbers on every slide except the title slide
- [ ] Spell-checked
- [ ] Every borrowed figure, number, method or dataset is cited (`[1] F. First … L. Last. Title. Journal (Year).`)
- [ ] Error estimates (CI or SD) on **every** performance number
- [ ] Legends, axis labels and text large enough to read from the back of the room
- [ ] Dry run with a timer, within the time limit
- [ ] Everyone knows the main message and can explain every slide

## Style

- As little text as possible: at most about six short bullets, no lecture slides.
- Large fonts, for text and for plots. Nobody has ever complained that a font was too big.
- Colour only where it carries meaning: colour-blind safe (we use TUM blue `#0065BD`, orange `#E37222` and grey), consistent across slides, and still readable in greyscale. No rainbow maps, no 3D, no decoration.
- Show the data (Tufte): high data-to-ink ratio, sort bars by size, use lollipops instead of many bars, no Venn diagrams (use bar or UpSet plots). Pick plot types with [data-to-viz.com](https://www.data-to-viz.com).
- Mark the point of each plot with an arrow or highlight, and make slide titles state the message.

## Content

- Never oversell. Give a realistic, if need be conservative, interpretation. Grading rewards scientific soundness, including scepticism about our own results, not high accuracy.
- Discuss unexpected findings openly.
- Credit people we worked with (e.g. the CB group on the same project).
