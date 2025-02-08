provider "aws" {
  region = var.region
}

resource "aws_ecs_task_definition" "ecs_task_definition" {
 family             = "StoryGenerationServiceTest"
 network_mode       = "awsvpc"
 execution_role_arn = "arn:aws:iam::533047435524:role/ecsTaskExecutionRole"
 cpu                = 256
 requires_compatibilities = ["EC2", "FARGATE"]
 runtime_platform {
   operating_system_family = "LINUX"
   cpu_architecture        = "X86_64"
 }
 container_definitions = jsonencode([
   {
     name      = "StoryGenerationServiceTest"
     image     = "public.ecr.aws/ecs-sample-image/amazon-ecs-sample:latest"
     cpu       = 256
     memory    = 512
     essential = true
     portMappings = [
       {
         containerPort = 80
         hostPort      = 80
         protocol      = "tcp"
       }
     ]
   }
 ])
}