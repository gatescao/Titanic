library(shiny)
library(tree)
library(dplyr)
library(rpart)
library(caret)

#Build model
tree_fit <- tree(as.factor(Survived) ~., data = training)

ui <- fluidPage(
  titlePanel("Titanic Survival Prediction"),
  sidebarLayout(
    sidebarPanel(
      sliderInput("Sex",
                  "Sex",
                  min = min(training$Sex, na.rm = TRUE),
                  max = max(training$Sex, na.rm = TRUE),
                  step = 1,
                  value = 0),
      sliderInput("Pclass",
                  "Pclass",
                  min = min(training$Pclass, na.rm = TRUE),
                  max = max(training$Pclass, na.rm = TRUE),
                  step = 1,
                  value = 3),
      sliderInput("Embarked",
                  "Embarked",
                  min = min(training$Embarked, na.rm = TRUE),
                  max = max(training$Embarked, na.rm = TRUE),
                  step = 1,
                  value = 0),
      sliderInput("Age",
                  "Age",
                  min = min(training$Age, na.rm = TRUE),
                  max = max(training$Age, na.rm = TRUE),
                  step = 0.5,
                  value = 15),
      sliderInput("SibSp",
                  "Number of siblings and spouse",
                  min = min(training$SibSp, na.rm = TRUE),
                  max = max(training$SibSp, na.rm = TRUE),
                  step = 1,
                  value = 1),
      sliderInput("Parch",
                  "Number of parents and children",
                  min = min(training$Parch, na.rm = TRUE),
                  max = max(training$Parch, na.rm = TRUE),
                  step = 1,
                  value = 1),
      sliderInput("Fare",
                  "Fare",
                  min = min(training$Fare, na.rm = TRUE),
                  max = max(training$Fare, na.rm = TRUE),
                  step = 0.01,
                  value = 26)
    ),
    mainPanel(
      textOutput("prediction"),
      plotOutput("tree")
    )
  )
)

server <- function(input, output) {
  generate_reactive <- reactive({
    tree(as.factor(Survived) ~ ., data = training)
  })
  output$prediction <- renderText({
    new_data = data.frame(
      Sex = input$Sex,
      Pclass = input$Pclass,
      Embarked = input$Embarked,
      Age = input$Age,
      SibSp = input$SibSp,
      Parch = input$Parch,
      Fare = input$Fare
    )
    titanic_predict <- predict(generate_reactive(), new_data, type = "class")
    titanic_predict <- ifelse(titanic_predict == 0, "died", "survived" )
    paste("This passenger", titanic_predict, ".", sep = " ")
  })
  output$tree <- renderPlot({
    plot(generate_reactive())
    text(generate_reactive())
    #fancyRpartPlot(titanic.predict)
  })
}

shinyApp(ui = ui, server = server)
