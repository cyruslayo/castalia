# Adaptive AI Math Learning Playground — Technical & Architectural Design Document

## 1. Executive Summary

This document describes the architecture for an AI-powered adaptive mathematics learning application inspired by the interactive, scenario-based learning style associated with Synthesis and the SpaceX/Ad Astra school tradition.

The product should not behave like a generic chatbot that simply explains math. Instead, it should function as an **AI-orchestrated interactive learning environment** where students learn by manipulating objects, solving puzzles, making choices, receiving adaptive feedback, and progressing through carefully designed learning scenarios.

The central design principle is:

> The AI agent should not invent lessons freely. It should assemble, adapt, and orchestrate structured learning scenarios from a controlled curriculum, misconception, scenario, and playground-component library.

The system combines:

- A structured curriculum graph
- A scenario-template library
- An interactive math playground
- A student model
- A misconception engine
- A mastery engine
- A deterministic runtime
- An AI tutor/orchestrator
- A data-generation and evaluation pipeline
- Optional model fine-tuning later, once real usage data exists

The first recommended MVP domain is **Grade 3 fractions**, especially:

1. Equal parts
2. Unit fractions
3. Comparing unit fractions
4. Fraction bars and visual comparison
5. Common misconceptions such as “larger denominator means larger fraction”

---

## 2. Product Vision

### 2.1 What the App Is

The app is an adaptive mathematics learning playground where a student interacts with visual math objects while an AI tutor guides the experience.

A typical interaction may look like this:

1. The AI says: “You are a mission engineer. Two rockets have different fuel tanks. Choose the one with more fuel.”
2. The playground shows two fraction bars: `1/2` and `1/3`.
3. The student drags one tank into the rocket slot.
4. The system checks the action.
5. If the student chooses `1/3`, the system detects a likely misconception: “larger denominator means larger fraction.”
6. The AI gives short, encouraging feedback.
7. The playground overlays the bars and highlights piece size.
8. The next task adapts to the student’s need.

### 2.2 What the App Is Not

The app should not be:

- A worksheet generator
- A generic chatbot tutor
- A passive video lesson platform
- A collection of disconnected quizzes
- A fully freeform AI lesson generator
- A math game with no serious student model

The system should combine **pedagogy, interactivity, adaptive sequencing, and AI dialogue**.

---

## 3. Core Design Principles

### 3.1 Scenario-Based Learning

The system should teach through structured scenarios rather than isolated questions.

Example:

Instead of:

> What is larger: 1/2 or 1/3?

Use:

> Two rockets have fuel tanks of the same size. Rocket A has 1/2 tank of fuel. Rocket B has 1/3 tank of fuel. Drag the rocket with more fuel into the launch zone.

This makes the student reason through a meaningful visual situation.

### 3.2 Interactive Manipulation

Students should act on the environment:

- Drag
- Drop
- Split
- Sort
- Match
- Build
- Count
- Compare
- Overlay
- Place on a number line
- Repair a wrong model

Every action should create evidence about the student’s thinking.

### 3.3 Misconception-Driven Adaptation

The system should not only know that an answer is wrong. It should identify why the student may be wrong.

Example:

Wrong action:

> Student chooses 1/8 as larger than 1/3.

Likely misconception:

> The student thinks a larger denominator means a larger fraction.

Recommended intervention:

> Show same-sized wholes, highlight one part from each, and overlay the pieces.

### 3.4 Controlled AI Agent

The AI agent should work through tools and schemas. It should not directly manipulate arbitrary UI or invent uncontrolled activities.

The AI should produce structured scene instructions such as:

```json
{
  "tool": "create_playground_scene",
  "arguments": {
    "template": "fraction_bar_compare",
    "fractions": ["1/2", "1/3"],
    "task": "Drag the larger amount of fuel to the rocket.",
    "misconception_target": "larger_denominator_means_larger_fraction"
  }
}
```

The deterministic runtime should render this safely.

### 3.5 Deterministic Runtime, Adaptive Intelligence

The runtime should be deterministic and validated. The AI can decide or suggest the next learning move, but the system should validate:

- Schema correctness
- Curriculum boundaries
- Answer checking
- Safety constraints
- Grade-level appropriateness
- Allowed tool calls
- Student data privacy

---

## 4. High-Level System Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                         Student App                         │
│                                                             │
│  ┌─────────────────┐    ┌───────────────────────────────┐   │
│  │ Tutor Chat UI   │    │ Interactive Learning Playground│   │
│  │                 │    │                               │   │
│  │ AI conversation │    │ Canvas, manipulatives, games   │   │
│  └─────────────────┘    └───────────────────────────────┘   │
│             │                         │                     │
└─────────────┼─────────────────────────┼─────────────────────┘
              │                         │
              ▼                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Learning Runtime Engine                  │
│                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │ Scenario Player │  │ Event Collector │  │ Feedback UI │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│                       AI Agent Layer                        │
│                                                             │
│  ┌────────────────────┐ ┌────────────────────┐             │
│  │ Tutor Orchestrator │ │ Scenario Generator │             │
│  └────────────────────┘ └────────────────────┘             │
│                                                             │
│  ┌────────────────────┐ ┌────────────────────┐             │
│  │ Misconception Agent│ │ Feedback Agent     │             │
│  └────────────────────┘ └────────────────────┘             │
└─────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Learning Intelligence                    │
│                                                             │
│  ┌──────────────────┐ ┌──────────────────┐ ┌─────────────┐ │
│  │ Student Model    │ │ Mastery Engine   │ │ Policy Engine│ │
│  └──────────────────┘ └──────────────────┘ └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Content System                         │
│                                                             │
│  ┌──────────────────┐ ┌──────────────────┐ ┌─────────────┐ │
│  │ Curriculum Graph │ │ Scenario Library │ │ Tool Library│ │
│  └──────────────────┘ └──────────────────┘ └─────────────┘ │
│                                                             │
│  ┌──────────────────┐ ┌──────────────────┐                 │
│  │ Misconception DB │ │ Assessment Bank  │                 │
│  └──────────────────┘ └──────────────────┘                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. Major System Layers

## 5.1 Student App Layer

The student-facing app has two primary interfaces:

1. Tutor Chat UI
2. Interactive Learning Playground

### 5.1.1 Tutor Chat UI

The Tutor Chat UI handles natural language interaction. It should be warm, concise, and focused on guiding the activity.

Example messages:

- “Let’s compare these two fuel tanks.”
- “Good try. Let’s check the size of each piece.”
- “Now build the fraction yourself.”
- “You solved three in a row. Let’s try a harder one.”

The chat should not dominate the learning experience. It should support the playground.

### 5.1.2 Interactive Learning Playground

The playground is the main teaching environment.

Recommended frontend stack:

- React
- TypeScript
- Next.js
- tldraw or custom canvas layer
- Custom math manipulatives
- JSXGraph later for graphs, geometry, coordinate planes, and number lines

The playground should support:

- Fraction bars
- Fraction circles
- Number lines
- Counters
- Arrays
- Ten frames
- Base-ten blocks
- Place-value charts
- Geometry boards
- Drop zones
- Highlight overlays
- Guided animations

---

## 5.2 Learning Runtime Engine

The runtime engine executes the lesson scenarios.

It should be deterministic, validated, and independent of the LLM for basic execution.

```text
Learning Runtime Engine
├── Scenario Player
├── Canvas Renderer
├── Interaction Handler
├── Event Collector
├── Answer Checker
├── Hint Controller
└── Feedback Renderer
```

### 5.2.1 Scenario Player

The Scenario Player receives a scenario JSON object and renders the correct playground scene.

Responsibilities:

- Load scenario definition
- Validate schema
- Instantiate playground components
- Register allowed interactions
- Connect assessment rules
- Trigger feedback states
- Emit student events

### 5.2.2 Canvas Renderer

The renderer turns structured component configs into UI objects.

Example:

```json
{
  "id": "tank_A",
  "type": "fraction_bar",
  "x": 120,
  "y": 160,
  "props": {
    "numerator": 1,
    "denominator": 2,
    "whole_size": "same",
    "draggable": true,
    "label": "1/2"
  }
}
```

### 5.2.3 Interaction Handler

The Interaction Handler tracks user actions such as:

- Drag start
- Drag end
- Drop target
- Object selected
- Object grouped
- Fraction built
- Answer submitted
- Hint requested
- Explanation typed

### 5.2.4 Answer Checker

Answer checking should be deterministic. Do not rely on the LLM to decide whether a mathematical action is correct.

Example:

```json
{
  "correct_action": {
    "type": "drop",
    "object_id": "tank_A",
    "target_id": "rocket_slot"
  }
}
```

### 5.2.5 Event Collector

Every student action should be captured as an event.

Example:

```json
{
  "event_type": "student_dragged_object",
  "object_id": "fuel_tank_1_3",
  "target_id": "rocket_slot",
  "is_correct": false,
  "timestamp": "2026-04-30T10:25:00Z",
  "skill": "compare_unit_fractions",
  "suspected_misconception": "larger_denominator_means_larger_fraction"
}
```

---

## 5.3 AI Agent Layer

The AI agent should act as a tutor orchestrator.

It should not be the sole teacher. The complete system teaches through:

- Curriculum graph
- Scenario library
- Interactive playground
- Student model
- Misconception engine
- Mastery engine
- Feedback policies
- AI dialogue

### 5.3.1 Recommended Agent Modules

```text
Tutor Orchestrator
├── Curriculum Planner
├── Scenario Assembler
├── Misconception Diagnoser
├── Feedback Generator
├── Hint Generator
├── Assessment Generator
└── Reflection Summarizer
```

### 5.3.2 Tutor Orchestrator

The Tutor Orchestrator is the central coordinator.

Responsibilities:

- Read current student state
- Interpret recent events
- Select the next action
- Decide whether to remediate, continue, enrich, or advance
- Call the correct tools
- Keep the interaction coherent

### 5.3.3 Scenario Assembler

The Scenario Assembler creates a valid scenario instance from:

- Target skill
- Student profile
- Scenario template
- Misconception target
- Difficulty level
- Preferred context
- Available playground components

### 5.3.4 Misconception Diagnoser

The Misconception Diagnoser maps student errors to likely misconceptions.

Example:

```json
{
  "wrong_action": "selected_1_8_over_1_3",
  "likely_misconception": "larger_denominator_means_larger_fraction",
  "confidence": 0.81,
  "recommended_intervention": "equal_whole_fraction_bar_overlay"
}
```

### 5.3.5 Feedback Generator

The Feedback Generator produces age-appropriate, concise feedback based on a chosen feedback strategy.

It should not invent the pedagogy. It should phrase a known strategy warmly.

Example input:

```json
{
  "age": 8,
  "mistake": "selected_1_3_over_1_2",
  "feedback_strategy": "compare_piece_size",
  "tone": "encouraging"
}
```

Example output:

```text
Good try. You noticed the 3, but let’s look at the size of each piece. When the same whole is split into 3 parts, each part is smaller than when it is split into 2 parts.
```

---

## 6. Agent Tool Design

The AI agent should operate through controlled tools.

### 6.1 Core Agent Tools

```text
select_next_skill()
diagnose_misconception()
select_learning_scenario()
adapt_learning_scenario()
create_playground_scene()
generate_hint()
generate_feedback()
create_micro_assessment()
update_student_model()
summarize_learning_session()
```

### 6.2 Important Tool: `select_learning_scenario()`

Example call:

```json
{
  "student_id": "stu_001",
  "grade_level": "grade_3",
  "target_skill": "compare_unit_fractions",
  "mastery_score": 0.42,
  "misconceptions": [
    "larger_denominator_means_larger_fraction"
  ],
  "preferred_contexts": ["space", "games"],
  "interaction_mode": "drag_and_compare"
}
```

Example return:

```json
{
  "scenario_id": "fraction_space_fuel_001",
  "title": "Rocket Fuel Fractions",
  "objective": "Compare unit fractions using equal wholes.",
  "playground_template": "fraction_bar_compare",
  "story_context": "Two rockets have different fuel tanks. Choose the one with more fuel.",
  "objects": [
    {
      "type": "fraction_bar",
      "id": "tank_A",
      "parts": 2,
      "filled": 1,
      "label": "1/2"
    },
    {
      "type": "fraction_bar",
      "id": "tank_B",
      "parts": 3,
      "filled": 1,
      "label": "1/3"
    }
  ],
  "correct_action": {
    "type": "drag",
    "object_id": "tank_A",
    "target_id": "rocket_slot"
  },
  "feedback_rules": [
    {
      "condition": "student_selects_tank_B",
      "misconception": "larger_denominator_means_larger_fraction",
      "response_strategy": "show_equal_whole_overlay"
    }
  ]
}
```

### 6.3 Playground Control Tools

The agent should use component-level actions, not raw pixel drawing.

Recommended tool actions:

```text
create_object()
move_object()
highlight_object()
lock_object()
unlock_object()
compare_objects()
animate_partition()
show_hint_overlay()
show_equal_whole_overlay()
reset_scene()
create_fraction_bar()
create_number_line()
create_array_model()
create_drop_zone()
```

Example:

```json
{
  "tool": "create_fraction_bar",
  "arguments": {
    "id": "bar_A",
    "numerator": 1,
    "denominator": 3,
    "wholeSize": "same",
    "draggable": true,
    "showLabel": true
  }
}
```

---

## 7. Content System

The content system is the heart of the product.

```text
Content System
├── Curriculum Graph
├── Skill Graph
├── Lesson Templates
├── Scenario Templates
├── Misconception Library
├── Playground Component Library
├── Assessment Bank
└── Feedback Strategies
```

The system should not begin with thousands of lessons. Start with a small, high-quality content graph and expand gradually.

---

## 8. Curriculum Graph

The curriculum graph defines the progression of learning.

Example:

```json
{
  "skill_id": "compare_unit_fractions",
  "name": "Compare unit fractions",
  "grade_level": "3",
  "prerequisites": [
    "understand_equal_parts",
    "identify_unit_fractions",
    "same_whole_principle"
  ],
  "next_skills": [
    "compare_non_unit_fractions",
    "place_fractions_on_number_line"
  ],
  "common_misconceptions": [
    "larger_denominator_means_larger_fraction",
    "ignores_same_whole",
    "counts_parts_without_comparing_size"
  ]
}
```

### 8.1 Example Fraction Skill Graph

```text
Math
└── Fractions
    ├── Equal parts
    ├── Unit fractions
    ├── Same whole principle
    ├── Compare unit fractions
    ├── Compare non-unit fractions
    ├── Equivalent fractions
    └── Fractions on a number line
```

Each skill should include:

- Prerequisites
- Common misconceptions
- Scenario templates
- Recommended manipulatives
- Micro-assessments
- Mastery criteria
- Remediation paths
- Enrichment paths

---

## 9. Scenario Library

The scenario library is the internal equivalent of a Synthesis-like lesson deck.

A scenario is not just a question. It is an interactive learning pattern.

### 9.1 Scenario Template Example

```json
{
  "template_id": "fraction_bar_compare",
  "skill": "compare_unit_fractions",
  "interaction_type": "drag_and_compare",
  "pedagogical_goal": "Help the learner see that a larger denominator creates smaller equal parts when the whole is the same.",
  "objects_required": [
    "fraction_bar",
    "drop_zone",
    "highlight_overlay"
  ],
  "difficulty_variables": {
    "denominators": [2, 3, 4, 6, 8],
    "show_labels": true,
    "same_whole_visible": true,
    "include_distractor": false
  },
  "misconceptions_checked": [
    "larger_denominator_means_larger_fraction"
  ],
  "success_criteria": {
    "correct_attempts_required": 3,
    "max_hint_level": 1
  }
}
```

### 9.2 Difficulty Adaptation

Easy scenario:

```json
{
  "denominators": [2, 3],
  "show_labels": true,
  "same_whole_visible": true,
  "include_distractor": false
}
```

Harder scenario:

```json
{
  "denominators": [5, 8],
  "show_labels": false,
  "same_whole_visible": false,
  "include_distractor": true
}
```

---

## 10. Playground Component Library

The playground component library contains the reusable manipulatives that the agent can use.

### 10.1 Recommended Components

```text
fraction_bar
fraction_circle
number_line
array_model
ten_frame
base_ten_blocks
place_value_chart
counter_set
balance_scale
coordinate_grid
graph_plotter
geometry_board
measurement_ruler
angle_tool
pattern_blocks
drop_zone
highlight_overlay
comparison_overlay
```

### 10.2 Component API Example

```typescript
createFractionBar({
  id: "bar_A",
  numerator: 1,
  denominator: 3,
  wholeSize: "same",
  draggable: true,
  showLabel: true
})
```

### 10.3 Recommended First Components for MVP

For Grade 3 fractions MVP:

1. FractionBarShape
2. FractionCircleShape
3. DropZoneShape
4. HintOverlayShape
5. ComparisonOverlayShape

---

## 11. Misconception Library

The misconception library is essential for adaptive tutoring.

### 11.1 Example Misconception

```json
{
  "misconception_id": "larger_denominator_means_larger_fraction",
  "description": "Student thinks a fraction with a larger denominator is larger.",
  "diagnostic_patterns": [
    "chooses 1/8 over 1/3",
    "says 8 is bigger than 3",
    "counts number of parts instead of size of parts"
  ],
  "remediation_strategies": [
    "show equal wholes",
    "highlight one part size",
    "overlay pieces",
    "ask student to build both fractions"
  ]
}
```

### 11.2 Misconception Diagnosis Event

```json
{
  "wrong_action": "selected_1_8_over_1_3",
  "likely_misconception": "larger_denominator_means_larger_fraction",
  "confidence": 0.81,
  "recommended_intervention": "equal_whole_fraction_bar_overlay"
}
```

---

## 12. Assessment and Micro-Assessment System

Micro-assessments are small interactions that give evidence about student understanding.

### 12.1 Micro-Assessment Example

```json
{
  "assessment_id": "compare_unit_frac_001",
  "skill_id": "compare_unit_fractions",
  "prompt": "Drag the larger fraction into the box.",
  "objects": [
    {
      "type": "fraction_bar",
      "value": "1/2"
    },
    {
      "type": "fraction_bar",
      "value": "1/3"
    }
  ],
  "correct_action": {
    "type": "drag",
    "object": "1/2",
    "target": "larger_box"
  },
  "misconception_if_wrong": {
    "selected": "1/3",
    "tag": "larger_denominator_means_larger_fraction"
  }
}
```

### 12.2 Mastery Evidence

A single correct answer is not mastery.

The system should look for evidence across:

- Same representation
- Different representation
- Same story context
- Different story context
- Near-transfer tasks
- Reduced scaffolding
- Lower hint dependency

Example progression:

1. Student compares `1/2` and `1/3` using fraction bars.
2. Student compares `1/4` and `1/6` using pizza circles.
3. Student places `1/3` on a number line.
4. Student explains why `1/3` is greater than `1/6`.

---

## 13. Student Model

The student model stores what the system currently believes about the learner.

```text
Student Model
├── Skill mastery
├── Misconception history
├── Confidence level
├── Hint dependency
├── Response speed
├── Persistence/frustration signals
├── Preferred representations
└── Learning trajectory
```

### 13.1 Example Student Model

```json
{
  "student_id": "stu_001",
  "grade_level": "3",
  "skills": {
    "compare_unit_fractions": {
      "mastery": 0.42,
      "attempts": 8,
      "correct": 4,
      "last_seen": "2026-04-30",
      "active_misconceptions": [
        "larger_denominator_means_larger_fraction"
      ],
      "preferred_representation": "fraction_bar"
    }
  }
}
```

### 13.2 Initial Mastery Model

For MVP, use a simple weighted mastery score.

Potential inputs:

- Correctness
- Hint usage
- Time to response
- Number of attempts
- Correct streak
- Mistake type
- Transfer performance
- Representation diversity

Example logic:

```typescript
if (
  correctStreak >= 3 &&
  hintLevelUsed <= 1 &&
  transferScenarioCorrect === true
) {
  markSkillAsMastered(skillId)
  moveToNextSkill()
}
```

Later, the team may consider Bayesian Knowledge Tracing or another probabilistic mastery model.

---

## 14. Policy Engine

The policy engine decides what happens next.

It should be partly rule-based and partly AI-assisted.

### 14.1 Example Policy Rules

```typescript
if (student.frustration > 0.7) {
  return {
    action: "encourage_and_simplify",
    scenario_difficulty: "lower"
  }
}

if (diagnosis.misconception === "larger_denominator_means_larger_fraction") {
  return {
    action: "remediate",
    scenario_template: "equal_whole_fraction_overlay"
  }
}

if (mastery.score > 0.85 && transferCorrect) {
  return {
    action: "advance",
    next_skill: "compare_non_unit_fractions"
  }
}

return {
  action: "practice",
  scenario_template: "same_skill_new_context"
}
```

### 14.2 What Should Stay Rule-Based

The following should not depend solely on the model:

- Answer checking
- Mastery score calculation
- Curriculum prerequisite rules
- Safety rules
- Schema validation
- Allowed tool calls
- Student privacy rules
- Grade-level content boundaries

---

## 15. Agent Orchestration Loop

The heart of the product is the learning loop.

```text
1. Load student profile
2. Select target skill
3. Select scenario template
4. Render playground activity
5. Observe student interaction
6. Diagnose correctness/misconception
7. Update mastery model
8. Generate feedback
9. Adapt next step
10. Continue, remediate, enrich, or advance
```

### 15.1 Pseudocode

```typescript
async function learningTurn(studentId: string, event: StudentEvent) {
  const student = await StudentModel.get(studentId)

  const diagnosis = await MisconceptionEngine.diagnose({
    event,
    student
  })

  await StudentModel.update({
    studentId,
    diagnosis,
    event
  })

  const nextAction = await PolicyEngine.decide({
    student,
    diagnosis
  })

  if (nextAction.type === "feedback") {
    return FeedbackAgent.generate(nextAction)
  }

  if (nextAction.type === "new_scenario") {
    return ScenarioAgent.createScenario(nextAction)
  }

  if (nextAction.type === "hint") {
    return HintAgent.generate(nextAction)
  }
}
```

---

## 16. Scenario JSON Contract

The scenario JSON schema is the contract between:

- AI agent
- Content system
- Playground renderer
- Assessment engine
- Event system

### 16.1 Full Scenario Example

```json
{
  "scenario_id": "fraction_space_fuel_001",
  "version": "1.0",
  "title": "Rocket Fuel Fractions",
  "skill_id": "compare_unit_fractions",
  "objective": "Compare unit fractions using equal wholes.",
  "story": {
    "setting": "space mission",
    "student_role": "mission engineer",
    "goal": "choose the rocket with more fuel"
  },
  "playground": {
    "layout": "canvas",
    "components": [
      {
        "id": "tank_A",
        "type": "fraction_bar",
        "x": 120,
        "y": 160,
        "props": {
          "numerator": 1,
          "denominator": 2,
          "whole_size": "same",
          "draggable": true,
          "label": "1/2"
        }
      },
      {
        "id": "tank_B",
        "type": "fraction_bar",
        "x": 120,
        "y": 260,
        "props": {
          "numerator": 1,
          "denominator": 3,
          "whole_size": "same",
          "draggable": true,
          "label": "1/3"
        }
      },
      {
        "id": "rocket_slot",
        "type": "drop_zone",
        "x": 500,
        "y": 200,
        "props": {
          "accepts": ["tank_A"]
        }
      }
    ]
  },
  "assessment": {
    "correct_action": {
      "type": "drop",
      "object_id": "tank_A",
      "target_id": "rocket_slot"
    },
    "misconception_checks": [
      {
        "condition": {
          "type": "drop",
          "object_id": "tank_B",
          "target_id": "rocket_slot"
        },
        "misconception_id": "larger_denominator_means_larger_fraction"
      }
    ]
  },
  "feedback": {
    "correct": "Yes. One half is larger because the whole is split into fewer equal parts.",
    "incorrect": {
      "strategy": "visual_overlay",
      "message": "Good try. Let’s compare the size of one piece in each tank."
    }
  },
  "adaptation": {
    "if_correct": {
      "next": "increase_denominator_distance"
    },
    "if_wrong": {
      "next": "show_equal_whole_remediation"
    }
  }
}
```

---

## 17. Event System

Every student action should become an event.

```text
Student action → event → interpretation → student model update → next scenario
```

### 17.1 Example Event

```json
{
  "event_type": "object_dragged",
  "student_id": "stu_001",
  "session_id": "sess_009",
  "scenario_id": "fraction_space_fuel_001",
  "skill_id": "compare_unit_fractions",
  "object_id": "one_third_bar",
  "target_id": "larger_fraction_slot",
  "correct": false,
  "timestamp": "2026-04-30T14:30:00Z"
}
```

### 17.2 Event Types

Recommended events:

```text
session_started
scenario_loaded
object_selected
object_drag_started
object_dropped
answer_submitted
hint_requested
feedback_shown
student_explanation_submitted
scenario_completed
scenario_abandoned
skill_mastery_updated
misconception_detected
```

---

## 18. Backend Architecture

### 18.1 Recommended MVP Backend

```text
Frontend:
Next.js / React / TypeScript

Playground:
tldraw or custom canvas
Custom math components
JSXGraph later

Backend:
FastAPI or NestJS

Database:
PostgreSQL

Cache / sessions:
Redis

Event storage:
PostgreSQL event table for MVP
Later: Kafka, Redpanda, or dedicated analytics pipeline

LLM orchestration:
Gemma model behind an OpenAI-compatible interface
LangGraph or custom tool-calling orchestrator

Content storage:
JSON files initially
PostgreSQL JSONB later
Git-based content repository for versioning

Analytics:
Metabase, Superset, PostHog, or custom dashboard
```

### 18.2 Recommended MVP Stack

```text
Next.js
React
TypeScript
tldraw
FastAPI
PostgreSQL
Redis
Gemma 4 or equivalent model endpoint
JSON schema validation with Zod or Pydantic
```

---

## 19. API Design

### 19.1 Core API Endpoints

```text
POST /sessions/start
GET  /students/{student_id}/model
POST /scenario/next
POST /scenario/render
POST /events/log
POST /diagnosis/misconception
POST /feedback/generate
POST /mastery/update
POST /sessions/{session_id}/summary
```

### 19.2 Example `/scenario/next` Request

```json
{
  "student_id": "stu_001",
  "session_id": "sess_009",
  "current_skill": "compare_unit_fractions",
  "recent_events": [
    {
      "event_type": "object_dropped",
      "object_id": "one_third_bar",
      "target_id": "larger_fraction_slot",
      "correct": false
    }
  ]
}
```

### 19.3 Example `/scenario/next` Response

```json
{
  "next_action": "remediation_scenario",
  "scenario": {
    "scenario_id": "fraction_equal_whole_overlay_001",
    "skill_id": "compare_unit_fractions",
    "template_id": "equal_whole_fraction_overlay",
    "difficulty": "easy"
  }
}
```

---

## 20. Database Model

### 20.1 Core Tables

```text
users
students
skills
curriculum_edges
scenario_templates
scenario_instances
playground_components
student_events
student_skill_mastery
student_misconceptions
assessment_items
agent_messages
learning_sessions
```

### 20.2 Example SQL Schema

```sql
CREATE TABLE skills (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  grade_level TEXT,
  domain TEXT,
  description TEXT
);

CREATE TABLE curriculum_edges (
  id UUID PRIMARY KEY,
  from_skill_id TEXT REFERENCES skills(id),
  to_skill_id TEXT REFERENCES skills(id),
  relation_type TEXT
);

CREATE TABLE student_skill_mastery (
  student_id TEXT,
  skill_id TEXT,
  mastery_score NUMERIC DEFAULT 0,
  attempts INTEGER DEFAULT 0,
  correct_attempts INTEGER DEFAULT 0,
  last_practiced_at TIMESTAMP,
  PRIMARY KEY (student_id, skill_id)
);

CREATE TABLE student_events (
  id UUID PRIMARY KEY,
  student_id TEXT,
  session_id TEXT,
  scenario_id TEXT,
  skill_id TEXT,
  event_type TEXT,
  event_payload JSONB,
  correct BOOLEAN,
  created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE scenario_templates (
  id TEXT PRIMARY KEY,
  skill_id TEXT,
  template_type TEXT,
  title TEXT,
  template_json JSONB
);

CREATE TABLE student_misconceptions (
  id UUID PRIMARY KEY,
  student_id TEXT,
  skill_id TEXT,
  misconception_id TEXT,
  confidence NUMERIC,
  evidence JSONB,
  last_detected_at TIMESTAMP DEFAULT NOW()
);
```

---

## 21. Model Strategy With Gemma

The team plans to use Gemma 4 or a Gemma-family model.

The recommended approach is:

```text
Do not fine-tune first.
Build the controlled architecture first.
Generate structured content and eval data.
Use prompting, retrieval, tool-calling, and validation.
Fine-tune later only after observing failure patterns.
```

### 21.1 What the Model Should Do Initially

Gemma should act as:

```text
scenario assembler
dialogue tutor
hint generator
feedback generator
reflection summarizer
misconception explanation assistant
```

### 21.2 What the Model Should Not Control Alone

Gemma should not be solely responsible for:

```text
answer checking
mastery scoring
curriculum rules
student safety
allowed tool selection without validation
schema validity
privacy decisions
```

### 21.3 Model Routing

Use different model sizes or configurations for different workloads.

```text
Small model:
classification, simple hints, JSON repair, event summarization

Larger model:
scenario assembly, complex adaptation, reflective summaries, parent/teacher reports
```

---

## 22. Data Generation Strategy

Data generation is required from the start.

Fine-tuning is optional later.

### 22.1 Required Data Types

```text
1. Curriculum graph data
2. Skill prerequisite data
3. Misconception data
4. Scenario template data
5. Micro-assessment data
6. Feedback strategy data
7. Playground component configs
8. Student event logs
9. Tool-call examples
10. Evaluation datasets
```

### 22.2 Curriculum Graph Data Example

```json
{
  "skill_id": "compare_unit_fractions",
  "domain": "fractions",
  "grade": 3,
  "prerequisites": [
    "understand_equal_parts",
    "identify_unit_fractions",
    "same_whole_principle"
  ],
  "next_skills": [
    "compare_non_unit_fractions",
    "fractions_on_number_line"
  ]
}
```

### 22.3 Misconception Data Example

```json
{
  "misconception_id": "larger_denominator_means_larger_fraction",
  "skill_id": "compare_unit_fractions",
  "description": "Student thinks 1/8 is larger than 1/3 because 8 is larger than 3.",
  "diagnostic_patterns": [
    "selects 1/8 over 1/3",
    "says 8 is bigger than 3",
    "counts number of parts instead of comparing part size"
  ],
  "interventions": [
    "show_equal_whole_fraction_bars",
    "overlay_piece_sizes",
    "ask_student_to_build_each_fraction",
    "compare_same_whole_visually"
  ]
}
```

### 22.4 Tool-Call Dataset Example

Input:

```json
{
  "student_state": {
    "grade": 3,
    "target_skill": "compare_unit_fractions",
    "mastery": 0.38,
    "recent_errors": [
      {
        "selected": "1/8",
        "correct": "1/3",
        "context": "fraction_bar_compare"
      }
    ],
    "suspected_misconceptions": [
      "larger_denominator_means_larger_fraction"
    ]
  },
  "available_tools": [
    "select_scenario",
    "create_fraction_bar_scene",
    "show_equal_whole_overlay",
    "generate_hint",
    "update_student_model"
  ]
}
```

Expected agent action:

```json
{
  "tool": "select_scenario",
  "arguments": {
    "skill_id": "compare_unit_fractions",
    "scenario_type": "remediation",
    "template_id": "same_whole_fraction_bar_overlay",
    "difficulty": "easy",
    "misconception_target": "larger_denominator_means_larger_fraction",
    "fractions": ["1/2", "1/3"],
    "show_labels": true,
    "same_whole_visible": true
  }
}
```

---

## 23. Fine-Tuning Strategy

### 23.1 Fine-Tuning Is Not Required for MVP

The MVP should use:

```text
instruction-tuned Gemma
strong system prompts
retrieval from structured content
validated JSON schemas
tool calling
rule-based policy engine
evaluation suite
```

### 23.2 When Fine-Tuning Becomes Useful

Fine-tuning becomes useful when the team observes repeated model failures such as:

```text
wrong tool selection
invalid scenario JSON
poor misconception diagnosis
too much verbosity
weak age-appropriate feedback
bad difficulty adaptation
failure to follow product pedagogy
```

### 23.3 What to Fine-Tune For

Do not fine-tune the model to “know math.”

Fine-tune for product-specific behavior:

```text
tool selection
misconception diagnosis
scenario assembly
feedback style
structured JSON reliability
adaptive decision traces
```

### 23.4 Fine-Tuning Dataset Types

Recommended supervised examples:

1. Student state → next tool call
2. Wrong action → misconception diagnosis
3. Misconception → intervention strategy
4. Skill + profile → scenario JSON
5. Mistake + age + strategy → feedback text
6. Session events → learning summary

### 23.5 Suggested Fine-Tuning Path

```text
Stage 1: Prompting + retrieval + schemas
Stage 2: Collect failures and successful traces
Stage 3: Create supervised fine-tuning dataset
Stage 4: LoRA/QLoRA fine-tune a Gemma model
Stage 5: Evaluate against held-out tutoring cases
Stage 6: Deploy as specialized tutor-orchestrator model
Stage 7: Improve using real student outcome data
```

---

## 24. Evaluation Strategy

Before fine-tuning, build an evaluation suite.

### 24.1 What to Evaluate

```text
1. Valid JSON generation
2. Correct tool selection
3. Misconception diagnosis
4. Grade-appropriate explanation
5. Scenario quality
6. Adaptation quality
7. Safety and curriculum compliance
8. Conciseness of feedback
9. Correct use of available playground components
10. Consistency across sessions
```

### 24.2 Example Evaluation Item

```json
{
  "test_id": "frac_misconception_001",
  "input": "Student picked 1/8 over 1/3 and said 8 is bigger.",
  "expected": {
    "misconception": "larger_denominator_means_larger_fraction",
    "next_intervention": "same_whole_fraction_bar_overlay"
  }
}
```

### 24.3 Metrics

Possible metrics:

```text
schema_validity_rate
tool_selection_accuracy
misconception_classification_accuracy
feedback_length_score
age_appropriateness_score
curriculum_violation_rate
intervention_success_rate
next_attempt_improvement
hint_dependency_reduction
mastery_progression_rate
```

---

## 25. Synthetic Data Generation Plan

Before real student data exists, generate synthetic structured data.

### 25.1 Generate Scenario Variants

For each skill:

```text
10 story contexts
5 difficulty levels
3 interaction modes
5 misconception variants
3 remediation variants
```

For 20 skills, this can produce thousands of scenario configurations.

### 25.2 Generate Simulated Student Mistakes

Example:

```json
{
  "skill": "compare_unit_fractions",
  "student_profile": "denominator_confused",
  "likely_wrong_choices": [
    "choose larger denominator",
    "ignore equal whole",
    "count shaded pieces only"
  ]
}
```

### 25.3 Generate Agent Tool-Call Traces

Each trace should include:

```text
student state → diagnosis
diagnosis → tool choice
tool choice → scenario config
student response → feedback
feedback → next action
```

These traces serve two purposes:

1. Immediate evaluation data
2. Future fine-tuning data

---

## 26. MVP Scope

The recommended MVP should be narrow and high quality.

### 26.1 MVP Domain

```text
Grade 3 Fractions
```

### 26.2 MVP Skills

```text
1. Equal parts
2. Unit fractions
3. Same whole principle
4. Compare unit fractions
```

### 26.3 MVP Playground Components

```text
1. Fraction bar
2. Fraction circle
3. Drag-and-drop target
4. Highlight overlay
5. Comparison overlay
```

### 26.4 MVP Agent Tools

```text
1. select_learning_scenario
2. generate_feedback
3. adapt_difficulty
4. diagnose_misconception
5. update_student_model
```

### 26.5 MVP Content

Start with:

```text
20 handcrafted scenario templates
3–5 misconception types
30–50 micro-assessment variants
5 feedback strategies
1 simple mastery model
```

---

## 27. MVP Architecture

```text
Next.js App
│
├── Tutor Chat Panel
├── tldraw Playground
│   ├── FractionBarShape
│   ├── FractionCircleShape
│   ├── DropZoneShape
│   └── HintOverlayShape
│
├── Scenario Runtime
│   ├── renderScenario(json)
│   ├── captureEvents()
│   ├── checkAnswer()
│   └── sendEventToBackend()
│
└── API Backend
    ├── /scenario/next
    ├── /event/log
    ├── /student/model
    ├── /agent/feedback
    └── /agent/adapt
```

---

## 28. Build Roadmap

### Phase 1: Playground Prototype

Build:

```text
fraction bar
fraction circle
drag-and-drop answer
scenario JSON renderer
event logging
basic correctness check
```

Goal:

> Prove that structured JSON can render interactive math scenarios.

### Phase 2: Scenario Runtime

Build:

```text
scenario player
assessment rules
feedback rendering
hint levels
scenario completion logic
```

Goal:

> Prove that the runtime can execute lessons without relying on the LLM for correctness.

### Phase 3: Agent Scenario Selection

Build:

```text
student profile
skill mastery
select next scenario
generate feedback
adapt difficulty
```

Goal:

> Prove that the agent can orchestrate structured scenarios.

### Phase 4: Misconception Engine

Build:

```text
wrong answer patterns
misconception tagging
remediation scenario selection
hint-level adjustment
```

Goal:

> Prove that the app responds to why the student is wrong, not only whether they are wrong.

### Phase 5: Curriculum Graph

Build:

```text
skill prerequisites
next-skill recommendation
mastery thresholds
transfer tasks
```

Goal:

> Prove that learning progression is coherent and adaptive.

### Phase 6: Content Authoring System

Build:

```text
lesson template editor
scenario preview
assessment rule editor
misconception mapping editor
component config editor
```

Goal:

> Allow educators and designers to create high-quality scenario templates.

### Phase 7: Model Evaluation and Optional Fine-Tuning

Build:

```text
eval suite
failure log
supervised traces
LoRA/QLoRA experiments
held-out test set
model comparison dashboard
```

Goal:

> Improve model reliability only where prompting and validation are insufficient.

---

## 29. Engineering Design Choices

### 29.1 tldraw vs Custom Canvas

Use tldraw if the team wants faster canvas development, built-in object manipulation, custom shapes, and an infinite canvas.

Use a custom canvas if the team needs maximum control, lower-level rendering, or a more game-like environment.

Recommended MVP choice:

```text
tldraw + custom math shapes
```

### 29.2 JSXGraph

Use JSXGraph later for:

- Geometry
- Coordinate grids
- Graphing
- Function plotting
- Draggable points
- Number line explorations

For the first fractions MVP, custom React/tldraw components may be enough.

### 29.3 H5P

H5P can be useful as a reference for reusable activity formats and xAPI-style event tracking. However, it may be less suitable for a deeply agent-controlled custom playground.

### 29.4 OATutor-Inspired Adaptive Logic

OATutor-like ideas are useful for:

- Skill mastery tracking
- Knowledge tracing
- Adaptive sequencing
- Student model design

The team does not need to adopt OATutor directly, but its architecture is relevant.

---

## 30. Safety, Privacy, and Governance

Since the app is for education and potentially children, the system should include strict safeguards.

### 30.1 Safety Rules

```text
No open-ended unsafe chat
No uncontrolled agent UI manipulation
No unvalidated generated lessons
No grade-inappropriate content
No unsupported claims about student ability
No exposing internal student data in chat
```

### 30.2 Privacy Rules

```text
Minimize personal data collection
Store student data securely
Separate parent/teacher/admin access
Log model decisions for auditability
Avoid sending unnecessary student data to model endpoints
Provide deletion/export workflows
```

### 30.3 Content Governance

All scenario templates should be reviewed before production use.

AI-generated scenarios should pass:

- Schema validation
- Curriculum validation
- Safety validation
- Pedagogical validation
- Component availability validation

---

## 31. Key Risks and Mitigations

### Risk 1: The AI invents poor pedagogy

Mitigation:

- Use structured scenario templates
- Validate outputs
- Keep pedagogy in content library
- Use educator-reviewed templates

### Risk 2: The playground becomes too hard to build

Mitigation:

- Start with only fraction bars and circles
- Avoid full game engine initially
- Use tldraw or React components
- Build one domain deeply before expanding

### Risk 3: The model gives invalid JSON

Mitigation:

- Use strict schemas
- Add JSON repair
- Add retries
- Use constrained outputs where possible
- Evaluate schema validity

### Risk 4: The app becomes a quiz app, not a learning playground

Mitigation:

- Require each scenario to include manipulation
- Track misconception evidence
- Use feedback strategies
- Use visual remediation

### Risk 5: Fine-tuning happens too early

Mitigation:

- Build eval suite first
- Collect failure data
- Fine-tune only specific tasks
- Keep deterministic rules outside the model

---

## 32. Recommended Immediate Engineering Tasks

### 32.1 Week 1–2: Technical Spike

Build a prototype that can:

1. Render a fraction bar from JSON
2. Render two fraction bars and a drop zone
3. Capture drag-and-drop events
4. Check the answer deterministically
5. Display correct/incorrect feedback
6. Log student events

### 32.2 Week 3–4: Scenario Runtime

Build:

1. Scenario JSON schema
2. Scenario renderer
3. Assessment rule parser
4. Feedback renderer
5. Hint overlay
6. Basic student model

### 32.3 Week 5–6: Agent Integration

Build:

1. Tool-calling interface
2. Scenario selection endpoint
3. Feedback generation endpoint
4. Misconception diagnosis endpoint
5. Policy engine rules

### 32.4 Week 7–8: Content Expansion

Create:

1. 20 fraction scenarios
2. 3 misconception types
3. 30 micro-assessments
4. 5 remediation strategies
5. Basic teacher/admin review workflow

---

## 33. Final Recommended Architecture in One Sentence

The app should be built as:

> An AI-orchestrated adaptive tutoring system where the agent selects and adapts structured interactive math scenarios, the playground renders controlled manipulatives, the runtime captures student actions, the misconception engine interprets errors, and the student model determines what the learner needs next.

The AI is not the entire teacher.

The teacher is the full system:

```text
Curriculum graph
+ scenario library
+ interactive playground
+ student model
+ misconception engine
+ mastery engine
+ AI tutor dialogue
```

This is the architecture that gives the product a realistic path toward Synthesis-style adaptive learning while remaining buildable, testable, and safe for an engineering team.

