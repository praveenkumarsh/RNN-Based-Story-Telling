provider "aws" {
  region = var.region
}

# resource "aws_ecs_service" "this" {
#   name            = var.service_name
#   cluster         = var.cluster_id
#   desired_count   = var.desired_count

#   load_balancer {
#     # target_group_arn = aws_lb_target_group.foo.arn
#     container_name   = var.container_name
#     container_port   = var.container_port
#   }

#   launch_type = "EC2"
#   task_definition = var.ecs_task_definition_arn
#     # task_definition = "arn:aws:ecs:us-east-1:533047435524:task-definition/StoryGeneration:1"

    
# }

resource "aws_ecs_service" "this" {
  name            = var.service_name
  cluster         = var.cluster_id
  task_definition = "arn:aws:ecs:us-east-1:533047435524:task-definition/StoryGeneration:1"
  desired_count   = 1
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = aws_subnet.public[*].id
    security_groups  = [aws_security_group.ecs_security_group.id]
    assign_public_ip = true
  }
}